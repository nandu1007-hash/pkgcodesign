"""Linear thermo-elasticity with 8-node bricks + incompatible modes (Wilson-Taylor, 9 modes).

Why incompatible modes: package warpage is a *bending* response through only a few cells
in thickness. Plain Hex8 shear-locks and badly under-predicts warpage. The 9 internal
modes are statically condensed element-by-element (exact for rectangular bricks, which is
all the structured grid produces). Validated against the Timoshenko bi-material curvature
formula in tests/test_mechanical.py.

Units: SI (m, Pa, K). Voigt order [xx, yy, zz, yz, xz, xy] with engineering shear.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import StackGrid

_XI = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def _D(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def _B(xi, eta, zeta, h):
    """Compatible strain-displacement matrix (6x24) for a brick with half-sizes h=(hx,hy,hz)."""
    B = np.zeros((6, 24))
    for a, (xa, ya, za) in enumerate(_XI):
        # derivative wrt physical coords: dN/dx = (1/hx) dN/dxi, dN/dxi = xa(1+ya eta)(1+za zeta)/8
        dNdx = xa * (1 + ya * eta) * (1 + za * zeta) / 8 / h[0]
        dNdy = ya * (1 + xa * xi) * (1 + za * zeta) / 8 / h[1]
        dNdz = za * (1 + xa * xi) * (1 + ya * eta) / 8 / h[2]
        c = 3 * a
        B[0, c] = dNdx
        B[1, c + 1] = dNdy
        B[2, c + 2] = dNdz
        B[3, c + 1], B[3, c + 2] = dNdz, dNdy
        B[4, c], B[4, c + 2] = dNdz, dNdx
        B[5, c], B[5, c + 1] = dNdy, dNdx
    return B


def _G(xi, eta, zeta, h):
    """Incompatible-mode strain matrix (6x9): modes phi_m = (xi_m^2 - 1), 3 components each."""
    G = np.zeros((6, 9))
    pos = (xi, eta, zeta)
    for m in range(3):
        d = 2 * pos[m] / h[m]            # d phi_m / d x_m
        for comp in range(3):            # displacement component carrying this mode
            col = 3 * m + comp
            if comp == m:
                G[comp, col] = d                      # normal strain
            else:
                # shear strain gamma_{m,comp} = d u_comp/d x_m
                row = {(1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}[(m, comp)]
                G[row, col] = d
    return G


def element_matrices(h, E, nu):
    """Condensed stiffness Ke (24x24), unit-thermal load fth (24), centroid B (6x24), D (6x6)."""
    D = _D(E, nu)
    vol_w = h[0] * h[1] * h[2]            # det(J), weight 1 per Gauss point
    Kuu = np.zeros((24, 24)); Kua = np.zeros((24, 9)); Kaa = np.zeros((9, 9))
    fth = np.zeros(24)
    e_th = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    for xi in _GP:
        for eta in _GP:
            for zeta in _GP:
                B = _B(xi, eta, zeta, h); G = _G(xi, eta, zeta, h)
                Kuu += B.T @ D @ B * vol_w
                Kua += B.T @ D @ G * vol_w
                Kaa += G.T @ D @ G * vol_w
                fth += B.T @ D @ e_th * vol_w
    Ke = Kuu - Kua @ np.linalg.solve(Kaa, Kua.T)
    # thermal load on incompatible dofs integrates to zero for uniform dT per element
    return Ke, fth, _B(0.0, 0.0, 0.0, h), D


def solve_mechanical(grid: StackGrid, T: np.ndarray, t_ref: float, pin: str = "321"):
    """Thermal-stress solve for a cell-wise temperature field T [deg C] relative to t_ref.

    Free-standing package: 3-2-1 point constraints on the bottom face remove rigid-body motion.
    Returns dict: u (nx+1,ny+1,nz+1,3), von_mises (nx,ny,nz) [Pa], warpage_m, bottom_uz.
    """
    nx, ny, nz = grid.shape
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1
    nn = nnx * nny * nnz

    E = grid.cell_property(lambda m: m.E_Pa).ravel()
    nu = grid.cell_property(lambda m: m.nu).ravel()
    alpha = grid.cell_property(lambda m: m.alpha).ravel()
    dT = (np.asarray(T, dtype=float) - t_ref).ravel()

    I, J, Kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    I, J, Kk = I.ravel(), J.ravel(), Kk.ravel()
    nid = lambda i, j, k: (i * nny + j) * nnz + k
    corner_nodes = np.stack([nid(I + int((c[0] + 1) / 2), J + int((c[1] + 1) / 2), Kk + int((c[2] + 1) / 2))
                             for c in _XI], axis=1)                       # (nel, 8)
    dofs = (3 * corner_nodes[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)

    # unique element types
    hx = np.round(grid.dx[I] / 2, 12); hy = np.round(grid.dy[J] / 2, 12); hz = np.round(grid.dz[Kk] / 2, 12)
    keys = np.stack([hx, hy, hz, E, nu], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    cache = [element_matrices(u[:3], u[3], u[4]) for u in uniq]
    Ke_u = np.array([c[0] for c in cache]); fth_u = np.array([c[1] for c in cache])
    Bc_u = np.array([c[2] for c in cache]); D_u = np.array([c[3] for c in cache])

    Ke = Ke_u[inv]
    rows = np.repeat(dofs, 24, axis=1).ravel()
    cols = np.tile(dofs, (1, 24)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(3 * nn, 3 * nn)).tocsr()

    fe = fth_u[inv] * (alpha * dT)[:, None]
    # vectorised scatter (np.add.at is ~10-50x slower and was the assembly hot spot)
    f = np.bincount(dofs.ravel(), weights=fe.ravel(), minlength=3 * nn)

    # 3-2-1 constraints at bottom (k=0)
    A, B_, C = nid(0, 0, 0), nid(nx, 0, 0), nid(0, ny, 0)
    fixed = [3 * A, 3 * A + 1, 3 * A + 2, 3 * B_ + 1, 3 * B_ + 2, 3 * C + 2]
    free = np.setdiff1d(np.arange(3 * nn), fixed)
    u = np.zeros(3 * nn)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])

    # post-processing: von Mises at element centroids
    ue = u[dofs]
    eps = np.einsum("eij,ej->ei", Bc_u[inv], ue) - (alpha * dT)[:, None] * np.array([1, 1, 1, 0, 0, 0])
    sig = np.einsum("eij,ej->ei", D_u[inv], eps)
    sxx, syy, szz, syz, sxz, sxy = sig.T
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                 + 3 * (syz ** 2 + sxz ** 2 + sxy ** 2))

    u3 = u.reshape(nnx, nny, nnz, 3)
    bottom_uz = u3[:, :, 0, 2]
    X, Y = np.meshgrid(grid.x_edges, grid.y_edges, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
    coef, *_ = np.linalg.lstsq(P, bottom_uz.ravel(), rcond=None)
    resid = bottom_uz.ravel() - P @ coef
    return {"u": u3, "von_mises": vm.reshape(grid.shape), "warpage_m": float(resid.max() - resid.min()),
            "warpage_signed_m": float(resid[np.argmax(np.abs(resid))]), "bottom_uz": bottom_uz,
            "bottom_uz_residual": resid.reshape(nnx, nny)}
