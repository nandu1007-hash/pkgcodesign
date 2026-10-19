"""Command line: `pkgcodesign run model.json --out results.h5 [--reflow]`, `validate`, `rom`."""
import argparse
import json

from .boundary import BoundaryConditions
from .coupling import solve_coupled, solve_reflow
from .io_formats import load_package_model, merge_device_power, save_results_h5
from .materials import MaterialDB
from .mesh import build_grid
from .rom import ThermalROM


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pkgcodesign")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        p.add_argument("model")
        p.add_argument("--device-power", help="xtco.device_power.v1 JSON from Tracks 1/3")
        p.add_argument("--materials", help="custom materials JSON")

    r = sub.add_parser("run", help="run thermo-mechanical simulation")
    model_args(r)
    r.add_argument("--out", default="results.h5")
    r.add_argument("--reflow", action="store_true", help="solve reflow cool-down warpage instead")

    v = sub.add_parser("validate", help="validate a package-model JSON")
    v.add_argument("model")

    m = sub.add_parser("rom", help="train and validate the reduced-order thermal model")
    model_args(m)
    m.add_argument("--samples", type=int, default=60, help="offline snapshot count (default 60)")
    m.add_argument("--test", type=int, default=20, help="held-out validation count (default 20)")
    m.add_argument("--out", help="write the ROM report as JSON")

    a = ap.parse_args(argv)
    spec = load_package_model(a.model)
    if a.cmd == "validate":
        print("OK:", spec.get("name", a.model)); return 0

    if a.device_power:
        with open(a.device_power) as f:
            spec = merge_device_power(spec, json.load(f))
    db = MaterialDB.from_json(a.materials) if a.materials else MaterialDB.default()
    grid = build_grid(spec, db)

    if a.cmd == "rom":
        rom = ThermalROM(grid, spec["heat_sources"],
                          bc=BoundaryConditions.from_spec(spec)).train(n_samples=a.samples)
        rep = rom.validate(n_test=a.test)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(rep, f, indent=2)
        print(f"rank={rep['rank']}, max_err={rep['max_abs_err_K']:.2e} K, "
              f"speedup={rep['speedup']:.0f}x ({rep['n_dofs']} dofs)", "->", a.out or "stdout")
        return 0

    res = solve_reflow(grid, spec) if a.reflow else solve_coupled(grid, spec)
    save_results_h5(a.out, grid, res, spec)
    msg = f"warpage = {res['warpage_m']*1e6:.2f} um"
    if "T" in res and not a.reflow:
        msg += f", Tmax = {res['T'].max():.1f} C"
    print(msg, "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
