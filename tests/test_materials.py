"""Material database: declared units are enforced and out-of-range values are rejected (C3)."""
import json

import pytest

from pkgcodesign.materials import Material, MaterialDB


def _mat(**over):
    p = dict(name="m", k=148.0, cte=2.6, E=130.0, nu=0.28, rho=2329.0, cp=700.0)
    p.update(over)
    return Material(**p)


def test_shipped_database_loads_and_validates():
    db = MaterialDB.default()
    assert len(db.names()) == 8


def test_cte_given_per_kelvin_is_rejected():
    """Regression for C3: 2.6e-6 (1/K) used to pass the ppm/K range check unnoticed."""
    with pytest.raises(ValueError, match="cte"):
        MaterialDB().add(_mat(cte=2.6e-6))


def test_density_given_in_g_per_cm3_is_rejected():
    with pytest.raises(ValueError, match="rho"):
        MaterialDB().add(_mat(rho=2.33))


def test_youngs_modulus_given_in_pa_is_rejected():
    with pytest.raises(ValueError, match="E"):
        MaterialDB().add(_mat(E=130e9))


def test_plausible_values_are_accepted():
    MaterialDB().add(_mat())


def _db_dict(**units_over):
    units = {"k": "W/(m.K)", "cte": "ppm/K", "E": "GPa", "nu": "-", "rho": "kg/m3", "cp": "J/(kg.K)"}
    units.update(units_over)
    return {"schema": "xtco.materials.v1", "units": units,
            "materials": {"silicon": {"k": 148.0, "cte": 2.6, "E": 130.0, "nu": 0.28,
                                      "rho": 2329, "cp": 700, "source": "x", "confidence": "low"}}}


def test_declared_units_must_match_expected():
    with pytest.raises(ValueError, match="units"):
        MaterialDB.from_dict(_db_dict(cte="1/K"))


def test_wrong_schema_tag_is_rejected():
    d = _db_dict()
    d["schema"] = "xtco.materials.v0"
    with pytest.raises(ValueError, match="schema"):
        MaterialDB.from_dict(d)


def test_database_round_trip_keeps_units(tmp_path):
    db = MaterialDB.from_dict(_db_dict())
    p = tmp_path / "m.json"
    db.to_json(str(p))
    data = json.loads(p.read_text())
    assert data["units"]["cte"] == "ppm/K"
    assert MaterialDB.from_json(str(p)).get("silicon").cte == 2.6
