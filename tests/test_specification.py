from src.specification import build_dimension, tolerance_magnitudes, upsert_dimension


def test_build_symmetric_dimension():
    dim = build_dimension("A", 10.0, 0.2)
    assert dim["min"] == 9.8
    assert dim["max"] == 10.2
    assert dim["lower_tol"] == -0.2
    assert dim["upper_tol"] == 0.2
    assert dim["tolerance_type"] == "symmetric"


def test_build_asymmetric_and_upsert():
    dim = build_dimension("D", 4.75, 0.05, 0.0, diameter=True)
    assert dim["min"] == 4.7
    assert dim["max"] == 4.75
    assert tolerance_magnitudes(dim) == (0.05, 0.0)
    merged = upsert_dimension([{"label": "D", "nominal": 1.0}], dim)
    assert len(merged) == 1
    assert merged[0]["nominal"] == 4.75
    assert merged[0]["diameter"] is True
