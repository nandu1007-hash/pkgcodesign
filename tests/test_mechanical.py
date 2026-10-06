import numpy as np
from pkgcodesign import MaterialDB, build_grid
from pkgcodesign.materials import Material
from pkgcodesign.mechanical import solve_mechanical, element_matrices


def test_element_has_six_rigid_modes():
    Ke, *_ = element_matrices((1.0, 2.0, 0.5), 1.0, 0.3)
    w = np.linalg.eigvalsh(Ke)
    assert int((w < 1e-9 * w.max()).sum()) == 6


def test_free_expansion_is_stress_free():
    spec = {"domain_um": [200, 200], "mesh": {"nx": 4, "ny": 4},
            "layers": [{"name": "a", "material": "copper", "thickness_um": 100, "nz": 2}]}
    g = build_grid(spec)
    T = np.full(g.shape, 125.0)
    r = solve_mechanical(g, T, t_ref=25.0)
    assert r["von_mises"].max() < 1e-3 * 117e9 * 17e-6 * 100          # << constrained stress level
    # edge displacement = alpha * dT * L  (rigid-body motion removed by 3-2-1 at x=0 corner)
    assert np.isclose(r["u"][-1, 0, 0, 0], 17e-6 * 100 * 200e-6, rtol=1e-6)


def test_bimaterial_curvature_matches_timoshenko():
    """Two-layer plate, equal nu: warpage vs analytic bilayer curvature (biaxial moduli)."""
    from pkgcodesign import MaterialDB
    db = MaterialDB()
    nu = 0.3
    db.add(Material("A", k=1, cte=3.0, E=130, nu=nu, rho=1000, cp=1000))
    db.add(Material("B", k=1, cte=17.0, E=70, nu=nu, rho=1000, cp=1000))
    L, t1, t2, dT = 2000e-6, 100e-6, 100e-6, 150.0
    spec = {"domain_um": [L * 1e6, L * 1e6], "mesh": {"nx": 8, "ny": 8},
            "layers": [{"name": "a", "material": "A", "thickness_um": 100, "nz": 3},
                       {"name": "b", "material": "B", "thickness_um": 100, "nz": 3}]}
    g = build_grid(spec, db)
    r = solve_mechanical(g, np.full(g.shape, 25.0 + dT), t_ref=25.0)
    # Timoshenko with biaxial modulus E' = E/(1-nu)
    E1, E2 = 130e9 / (1 - nu), 70e9 / (1 - nu)
    m, n, h = t1 / t2, E1 / E2, t1 + t2
    kappa = 6 * (17e-6 - 3e-6) * dT * (1 + m) ** 2 / (h * (3 * (1 + m) ** 2 + (1 + m * n) * (m ** 2 + 1 / (m * n))))
    # sag of a plate with uniform curvature over the plate diagonal: w ~ kappa * r^2 / 2
    r_corner = L / np.sqrt(2)
    w_expected = kappa * r_corner ** 2 / 2 * 2          # peak-to-valley between centre and corner(s) plane-fit residual
    # compare in terms of centre-to-corner relative deflection from the solver
    uz = r["bottom_uz"]
    c = uz[4, 4]; corners = np.mean([uz[0, 0], uz[-1, 0], uz[0, -1], uz[-1, -1]])
    measured = abs(c - corners)
    expected = kappa * (L / np.sqrt(2)) ** 2 / 2
    assert abs(measured - expected) / expected < 0.08
