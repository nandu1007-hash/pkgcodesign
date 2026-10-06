import numpy as np
from pkgcodesign import MaterialDB, build_grid
from pkgcodesign.thermal import solve_thermal, boundary_conductance


def slab_spec(nz=8):
    return {"schema": "xtco.package_model.v1", "domain_um": [1000, 1000], "mesh": {"nx": 3, "ny": 3},
            "layers": [{"name": "slab", "material": "silicon", "thickness_um": 400, "nz": nz}],
            "heat_sources": []}


def test_energy_balance():
    g = build_grid(slab_spec())
    src = [{"layer": "slab", "cell_layer": "bottom", "rect_um": [200, 200, 700, 700], "power_W": 3.0}]
    T, _ = solve_thermal(g, src, h_top=5000, h_bottom=200, temperature_dependent=False)
    hA = boundary_conductance(g, 5000, 200)
    assert abs(np.sum(hA * (T.ravel() - 25.0)) - 3.0) < 1e-9


def test_1d_analytic_uniform_generation():
    """Uniform volumetric heating, adiabatic bottom, film on top: exact parabola at cell centres."""
    g = build_grid(slab_spec(nz=10))
    P, h = 2.0, 4000.0
    src = [{"layer": "slab", "cell_layer": "all", "rect_um": [0, 0, 1000, 1000], "power_W": P}]
    T, _ = solve_thermal(g, src, h_top=h, h_bottom=0.0, temperature_dependent=False)
    L, A, k = 400e-6, 1000e-6 ** 2, 148.0
    qv = P / (A * L)
    zc = 0.5 * (g.z_edges[1:] + g.z_edges[:-1])
    # film sits at the top cell centre, so the parabola is referenced to it
    T_exact = 25 + P / (h * A) + qv / (2 * k) * (zc[-1] ** 2 - zc ** 2)
    assert np.allclose(T[1, 1, :], T_exact, rtol=1e-10, atol=1e-10)


def test_temperature_dependent_k_runs_hotter():
    g = build_grid(slab_spec())
    src = [{"layer": "slab", "cell_layer": "bottom", "rect_um": [0, 0, 1000, 1000], "power_W": 0.5}]
    T_c, _ = solve_thermal(g, src, temperature_dependent=False)
    T_v, info = solve_thermal(g, src, temperature_dependent=True)
    assert T_v.max() > T_c.max() and info["residual"] < 1e-6


def test_nonconvergence_raises():
    """Regression for C5: the Picard loop used to return silently after max_iter."""
    import pytest
    from pkgcodesign.thermal import ConvergenceError
    g = build_grid(slab_spec())
    src = [{"layer": "slab", "cell_layer": "bottom", "rect_um": [0, 0, 1000, 1000], "power_W": 0.5}]
    with pytest.raises(ConvergenceError):
        solve_thermal(g, src, temperature_dependent=True, max_iter=1, tol=1e-12)


def test_info_reports_convergence():
    g = build_grid(slab_spec())
    src = [{"layer": "slab", "cell_layer": "bottom", "rect_um": [0, 0, 1000, 1000], "power_W": 0.5}]
    _, info = solve_thermal(g, src, temperature_dependent=True)
    assert info["converged"] is True and info["iterations"] >= 2 and info["residual"] < 1e-6
