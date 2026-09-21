import pandas as pd
import pytest

from src.features.pipeline import CATEGORY_OPTIONS
from src.utils.validate_data import validate_telco_data


@pytest.fixture
def raw_data():
    data = {column: [options[0], options[-1]] for column, options in CATEGORY_OPTIONS.items()}
    data.update({
        "customerID": ["1", "2"], "Churn": ["Yes", "No"],
        "SeniorCitizen": [0, 1], "tenure": [12, 0],
        "MonthlyCharges": [50.0, 0.0], "TotalCharges": ["600", " "],
    })
    return pd.DataFrame(data)


def test_valid_data_passes_without_mutation(raw_data):
    original = raw_data.copy(deep=True)
    assert validate_telco_data(raw_data) == (True, [])
    pd.testing.assert_frame_equal(raw_data, original)


@pytest.mark.parametrize("column", [
    "customerID", "gender", "Partner", "Dependents", "PhoneService",
    "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges",
])
def test_missing_required_column_fails(raw_data, column):
    success, failures = validate_telco_data(raw_data.drop(columns=column))
    assert not success
    assert "expect_column_to_exist" in failures


@pytest.mark.parametrize("column,value,expectation", [
    ("gender", "invalid", "expect_column_values_to_be_in_set"),
    ("tenure", -1, "expect_column_values_to_be_between"),
    ("MonthlyCharges", 201, "expect_column_values_to_be_between"),
    ("TotalCharges", "-1", "expect_column_values_to_be_between"),
    ("customerID", None, "expect_column_values_to_not_be_null"),
    ("TotalCharges", "invalid", "TotalCharges must contain numeric values or blanks"),
])
def test_invalid_values_fail(raw_data, column, value, expectation):
    raw_data.loc[0, column] = value
    success, failures = validate_telco_data(raw_data)
    assert not success
    assert expectation in failures


def test_inconsistent_charges_fail(raw_data):
    raw_data["TotalCharges"] = "0"
    success, failures = validate_telco_data(raw_data)
    assert not success
    assert "expect_column_pair_values_a_to_be_greater_than_b" in failures
