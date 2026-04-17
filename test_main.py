import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, BooleanType
from datetime import date
from dateutil.relativedelta import relativedelta
from main import add_age_and_eligibility_columns

# --- Constants for testing ---
MIN_DRIVING_AGE_TEST = 16
MIN_VOTING_AGE_TEST = 18

@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()

def test_add_age_and_eligibility_columns_can_drive_and_vote(spark):
    """
    Tests a person who is older than both the driving and voting age.
    """
    today = date.today()
    dob_over_18 = today - relativedelta(years=25)
    
    source_data = [("Alice", dob_over_18)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)
    
    actual_df = add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)
    
    expected_data = [("Alice", dob_over_18, 25, True, True)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)
    
    assert sorted(actual_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_can_drive_cannot_vote(spark):
    """
    Tests a person who is between the driving and voting age.
    """
    today = date.today()
    dob_17 = today - relativedelta(years=17)

    source_data = [("Bob", dob_17)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    actual_df = add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)

    expected_data = [("Bob", dob_17, 17, True, False)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    assert sorted(actual_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_cannot_drive_or_vote(spark):
    """
    Tests a person who is younger than both the driving and voting age.
    """
    today = date.today()
    dob_15 = today - relativedelta(years=15)

    source_data = [("Charlie", dob_15)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    actual_df = add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)

    expected_data = [("Charlie", dob_15, 15, False, False)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    assert sorted(actual_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_edge_case_driving_age_today(spark):
    """
    Tests a person whose 16th birthday is today. They should be eligible to drive.
    """
    today = date.today()
    dob_16_today = today - relativedelta(years=16)

    source_data = [("Dana", dob_16_today)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    actual_df = add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)

    expected_data = [("Dana", dob_16_today, 16, True, False)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    assert sorted(actual_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_edge_case_voting_age_today(spark):
    """
    Tests a person whose 18th birthday is today. They should be eligible to vote and drive.
    """
    today = date.today()
    dob_18_today = today - relativedelta(years=18)

    source_data = [("Eve", dob_18_today)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    actual_df = add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)

    expected_data = [("Eve", dob_18_today, 18, True, True)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)
    
    assert sorted(actual_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_edge_case_driving_age_tomorrow(spark):
    """
    Tests a person whose 16th birthday is tomorrow. They should not be eligible to drive yet.
    """
    today = date.today()
    dob_almost_16 = today - relativedelta(years=16) + relativedelta(days=1)

    source_data = [("Frank", dob_almost_16)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    actual_df = add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)

    expected_data = [("Frank", dob_almost_16, 15, False, False)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    assert sorted(actual_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_null_dob(spark):
    """
    Tests a record with a null 'dob'. Age should be null, and eligibility flags should be False.
    """
    source_data = [("Grace", None)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    actual_df = add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)

    expected_data = [("Grace", None, None, False, False)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    # Comparing nulls requires careful handling. collect() works fine.
    assert sorted(actual_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_preserves_other_columns(spark):
    """
    Tests that existing columns in the DataFrame are preserved.
    """
    today = date.today()
    dob_20 = today - relativedelta(years=20)
    
    source_data = [(101, "Heidi", "Customer", dob_20)]
    source_schema = StructType([
        StructField("id", IntegerType()),
        StructField("name", StringType()),
        StructField("type", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)
    
    actual_df = add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)
    
    expected_data = [(101, "Heidi", "Customer", dob_20, 20, True, True)]
    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("name", StringType()),
        StructField("type", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)
    
    assert sorted(actual_df.collect()) == sorted(expected_df.collect())
    assert "id" in actual_df.columns
    assert "type" in actual_df.columns

def test_add_age_and_eligibility_columns_with_multiple_rows(spark):
    """
    Tests the transformation on a DataFrame with multiple rows including duplicates and nulls.
    """
    today = date.today()
    dob_30 = today - relativedelta(years=30)
    dob_17 = today - relativedelta(years=17)
    dob_10 = today - relativedelta(years=10)

    source_data = [
        ("Ivan", dob_30),
        ("Judy", dob_17),
        ("Karl", dob_10),
        ("Judy", dob_17),  # Duplicate
        ("Liam", None)
    ]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    actual_df = add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)

    expected_data = [
        ("Ivan", dob_30, 30, True, True),
        ("Judy", dob_17, 17, True, False),
        ("Judy", dob_17, 17, True, False),
        ("Karl", dob_10, 10, False, False),
        ("Liam", None, None, False, False)
    ]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    assert sorted(actual_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the 'dob' column is missing.
    """
    source_data = [("Mallory",)]
    source_schema = StructType([StructField("name", StringType())])
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)
    
    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_eligibility_columns(source_df, MIN_DRIVING_AGE_TEST, MIN_VOTING_AGE_TEST)