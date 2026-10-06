"""Command line: `pkgcodesign run model.json --out results.h5 [--reflow]`"""
import argparse
import json

from .coupling import solve_coupled, solve_reflow
from .io_formats import load_package_model, merge_device_power, save_results_h5
from .materials import MaterialDB
from .mesh import build_grid


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pkgcodesign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run thermo-mechanical simulation")
    r.add_argument("model"); r.add_argument("--out", default="results.h5")
    r.add_argument("--device-power", help="xtco.device_power.v1 JSON from Tracks 1/3")
    r.add_argument("--materials", help="custom materials JSON")
    r.add_argument("--reflow", action="store_true", help="solve reflow cool-down warpage instead")
    v = sub.add_parser("validate", help="validate a package-model JSON")
    v.add_argument("model")
    a = ap.parse_args(argv)

    spec = load_package_model(a.model)
    if a.cmd == "validate":
        print("OK:", spec.get("name", a.model)); return 0
    if a.device_power:
        spec = merge_device_power(spec, json.load(open(a.device_power)))
    db = MaterialDB.from_json(a.materials) if a.materials else MaterialDB.default()
    grid = build_grid(spec, db)
    res = solve_reflow(grid, spec) if a.reflow else solve_coupled(grid, spec)
    save_results_h5(a.out, grid, res, spec)
    msg = f"warpage = {res['warpage_m']*1e6:.2f} um"
    if "T" in res and not a.reflow:
        msg += f", Tmax = {res['T'].max():.1f} C"
    print(msg, "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
