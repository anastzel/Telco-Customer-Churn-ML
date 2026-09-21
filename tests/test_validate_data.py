from pathlib import Path

import pandas as pd
import pytest

from src.utils.validate_data import validate_telco_data


@pytest.fixture
def raw_data():
    path = Path(__file__).resolve().parents[1] / "data/raw/Telco-Customer-Churn.csv"
    return pd.read_csv(path)


def test_raw_csv_passes_without_mutation(raw_data):
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
