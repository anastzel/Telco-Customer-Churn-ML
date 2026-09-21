import great_expectations as ge
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation with Great Expectations...")
    
    # Validate numeric charges without changing the raw input. Blank total
    # charges are allowed here and handled later by preprocessing.
    validation_df = df.copy()
    if "TotalCharges" in validation_df.columns:
        charges = validation_df["TotalCharges"].replace(r"^\s*$", None, regex=True)
        numeric_charges = pd.to_numeric(charges, errors="coerce")
        if (charges.notna() & numeric_charges.isna()).any():
            print("❌ Data validation FAILED: TotalCharges contains non-numeric values")
            return False, ["TotalCharges must contain numeric values or blanks"]
        validation_df["TotalCharges"] = numeric_charges

    # GX 1.x validates a DataFrame batch against an ExpectationSuite.
    context = ge.get_context(mode="ephemeral")
    context.variables.progress_bars = {"globally": False}
    asset = context.data_sources.add_pandas(name="telco").add_dataframe_asset(
        name="customers"
    )
    batch = asset.add_batch_definition_whole_dataframe("all_customers").get_batch(
        batch_parameters={"dataframe": validation_df}
    )
    expectations = []

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    # Customer identifier must exist (required for business operations)  
    expectations.append(ge.expectations.ExpectColumnToExist(column="customerID"))
    expectations.append(ge.expectations.ExpectColumnValuesToNotBeNull(column="customerID"))
    
    # Core demographic features
    expectations.append(ge.expectations.ExpectColumnToExist(column="gender"))
    expectations.append(ge.expectations.ExpectColumnToExist(column="Partner"))
    expectations.append(ge.expectations.ExpectColumnToExist(column="Dependents"))
    
    # Service features (critical for churn analysis)
    expectations.append(ge.expectations.ExpectColumnToExist(column="PhoneService"))
    expectations.append(ge.expectations.ExpectColumnToExist(column="InternetService"))
    expectations.append(ge.expectations.ExpectColumnToExist(column="Contract"))
    
    # Financial features (key churn predictors)
    expectations.append(ge.expectations.ExpectColumnToExist(column="tenure"))
    expectations.append(ge.expectations.ExpectColumnToExist(column="MonthlyCharges"))
    expectations.append(ge.expectations.ExpectColumnToExist(column="TotalCharges"))
    
    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    # Gender must be one of expected values (data integrity)
    expectations.append(ge.expectations.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]))
    
    # Yes/No fields must have valid values
    expectations.append(ge.expectations.ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"]))
    expectations.append(ge.expectations.ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"]))
    expectations.append(ge.expectations.ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"]))
    
    # Contract types must be valid (business constraint)
    expectations.append(ge.expectations.ExpectColumnValuesToBeInSet(
        column="Contract",
        value_set=["Month-to-month", "One year", "Two year"]
    ))
    
    # Internet service types (business constraint)
    expectations.append(ge.expectations.ExpectColumnValuesToBeInSet(
        column="InternetService",
        value_set=["DSL", "Fiber optic", "No"]
    ))
    
    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")
    
    # Tenure must be non-negative (business logic - can't have negative tenure)
    expectations.append(ge.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0))
    
    # Monthly charges must be positive (business logic - no free service)
    expectations.append(ge.expectations.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0))
    
    # Total charges should be non-negative (business logic)
    expectations.append(ge.expectations.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0))
    
    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")
    
    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    expectations.append(ge.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120))
    
    # Monthly charges should be within reasonable business range
    expectations.append(ge.expectations.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200))
    
    # No missing values in critical numeric features  
    expectations.append(ge.expectations.ExpectColumnValuesToNotBeNull(column="tenure"))
    expectations.append(ge.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))
    
    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")
    
    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    expectations.append(ge.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        ignore_row_if="either_value_is_missing",
        mostly=0.95  # Allow 5% exceptions for edge cases
    ))
    
    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    suite = ge.ExpectationSuite(name="telco_validation", expectations=expectations)
    results = batch.validate(suite)
    
    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r.expectation_config.type
            failed_expectations.append(expectation_type)
    
    # Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations
