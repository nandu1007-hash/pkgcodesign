"""Input validation: unknown keys and missing required sections must be rejected, not defaulted (C1)."""
import copy
import json
import os

import pytest

from pkgcodesign.io_formats import SchemaError, load_package_model, validate_package_model

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "..", "examples", "stacked_2tier_tsv.json")


def _spec():
    with open(MODEL) as f:
        return json.load(f)


def test_reference_model_is_valid():
    validate_package_model(_spec())


def test_typo_in_boundary_key_is_rejected():
    """Regression for C1: 'boundry' used to be ignored and the h_top=3000 default silently applied."""
    spec = _spec()
    spec["boundry"] = spec.pop("boundary")
    with pytest.raises(SchemaError, match="boundry"):
        validate_package_model(spec)


def test_unknown_key_inside_boundary_is_rejected():
    spec = _spec()
    spec["boundary"]["h_tp"] = spec["boundary"].pop("h_top")
    with pytest.raises(SchemaError, match="h_tp"):
        validate_package_model(spec)


def test_missing_boundary_is_rejected():
    spec = _spec()
    del spec["boundary"]
    with pytest.raises(SchemaError):
        validate_package_model(spec)


def test_missing_boundary_field_is_rejected():
    spec = _spec()
    del spec["boundary"]["h_bottom"]
    with pytest.raises(SchemaError):
        validate_package_model(spec)


def test_missing_thermomech_is_rejected():
    spec = _spec()
    del spec["thermomech"]
    with pytest.raises(SchemaError):
        validate_package_model(spec)


def test_unknown_top_level_key_is_rejected():
    spec = _spec()
    spec["heat_source"] = spec.pop("heat_sources")  # typo of a required key
    with pytest.raises(SchemaError):
        validate_package_model(spec)


def test_unknown_layer_key_is_rejected():
    spec = _spec()
    spec["layers"][0]["thicknes_um"] = spec["layers"][0].pop("thickness_um")
    with pytest.raises(SchemaError):
        validate_package_model(spec)


def test_non_positive_thickness_is_rejected():
    spec = _spec()
    spec["layers"][0]["thickness_um"] = 0
    with pytest.raises(SchemaError):
        validate_package_model(spec)


def test_load_package_model_applies_same_checks(tmp_path):
    spec = _spec()
    spec["boundary"]["t_ambinet_C"] = spec["boundary"].pop("t_ambient_C")
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(spec))
    with pytest.raises(SchemaError):
        load_package_model(str(p))
