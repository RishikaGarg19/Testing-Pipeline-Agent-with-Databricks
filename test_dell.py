import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
    IntegerType,
    BooleanType,
)
from datetime import date
from dateutil.relativedelta import relativedelta

from main import add_age_and_eligibility_columns

# This exact fixture is required by the problem description.
# It handles both local and Databricks (Spark Connect) environments.
@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_add_age_and_eligibility_for_adult(spark):
    """
    Tests that a person over 18 is correctly identified as an adult
    who can drive and vote.
    """
    # Arrange
    today = date.today()
    dob_25_years_ago = today - relativedelta(years=25)
    
    source_data = [("Alice", dob_25_years_ago)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    # Act
    actual_df = add_age_and_eligibility_columns(source_df)
    
    # Assert
    expected_data = [("Alice", dob_25_years_ago, 25, True, True)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType()),
        StructField("canVote", BooleanType())
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert actual_df.schema == expected_df.schema
    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_for_minor(spark):
    """
    Tests that a person under 18 is correctly identified as a minor
    who cannot drive or vote.
    """
    # Arrange
    today = date.today()
    dob_15_years_ago = today - relativedelta(years=15)
    
    source_data = [("Bob", dob_15_years_ago)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    # Act
    actual_df = add_age_and_eligibility_columns(source_df)
    
    # Assert
    expected_data = [("Bob", dob_15_years_ago, 15, False, False)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType()),
        StructField("canVote", BooleanType())
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert actual_df.schema == expected_df.schema
    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_boundary_case_exactly_18(spark):
    """
    Tests the boundary case where a person's 18th birthday is today.
    """
    # Arrange
    today = date.today()
    dob_18_years_ago = today - relativedelta(years=18)
    
    source_data = [("Charlie", dob_18_years_ago)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    # Act
    actual_df = add_age_and_eligibility_columns(source_df)
    
    # Assert
    expected_data = [("Charlie", dob_18_years_ago, 18, True, True)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType()),
        StructField("canVote", BooleanType())
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert actual_df.schema == expected_df.schema
    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_boundary_case_day_before_18(spark):
    """
    Tests the boundary case where a person's 18th birthday is tomorrow.
    """
    # Arrange
    today = date.today()
    dob_almost_18 = today - relativedelta(years=18) + relativedelta(days=1)
    
    source_data = [("Diana", dob_almost_18)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    # Act
    actual_df = add_age_and_eligibility_columns(source_df)
    
    # Assert
    expected_data = [("Diana", dob_almost_18, 17, False, False)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType()),
        StructField("canVote", BooleanType())
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert actual_df.schema == expected_df.schema
    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_with_null_dob(spark):
    """
    Tests that a null 'dob' results in a null 'age' and False for eligibility flags.
    """
    # Arrange
    source_data = [("Eve", None)]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    # Act
    actual_df = add_age_and_eligibility_columns(source_df)
    
    # Assert
    # The when(col("age") >= 18) evaluates to null when age is null,
    # which falls through to the otherwise(False) clause.
    expected_data = [("Eve", None, None, False, False)]
    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canDrive", BooleanType()),
        StructField("canVote", BooleanType())
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert actual_df.schema == expected_df.schema
    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_with_duplicate_rows(spark):
    """
    Tests that duplicate input rows are preserved in the output.
    """
    # Arrange
    today = date.today()
    dob = today - relativedelta(years=30)
    
    source_data = [
        ("Frank", dob),
        ("Frank", dob)
    ]
    source_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    # Act
    actual_df = add_age_and_eligibility_columns(source_df)
    
    # Assert
    expected_data = [
        ("Frank", dob, 30, True, True),
        ("Frank", dob, 30, True, True)
    ]
    expected_df = spark.createDataFrame(expected_data, spark.createDataFrame([], actual_df.schema).schema)
    
    # Sorting is important as DataFrame order is not guaranteed
    assert sorted(actual_df.collect(), key=lambda r: r.name) == sorted(expected_df.collect(), key=lambda r: r.name)


def test_add_age_and_eligibility_with_multiple_rows(spark):
    """
    Tests the transformation with a mix of different ages and nulls.
    """
    # Arrange
    today = date.today()
    dob_adult = today - relativedelta(years=40)
    dob_minor = today - relativedelta(years=10)
    
    source_data = [
        Row(id=1, name="Grace", dob=dob_adult),
        Row(id=2, name="Heidi", dob=dob_minor),
        Row(id=3, name="Ivan", dob=None)
    ]
    source_df = spark.createDataFrame(source_data)
    
    # Act
    actual_df = add_age_and_eligibility_columns(source_df)
    
    # Assert
    expected_data = [
        Row(id=1, name="Grace", dob=dob_adult, age=40, canDrive=True, canVote=True),
        Row(id=2, name="Heidi", dob=dob_minor, age=10, canDrive=False, canVote=False),
        Row(id=3, name="Ivan", dob=None, age=None, canDrive=False, canVote=False)
    ]
    expected_df = spark.createDataFrame(expected_data, spark.createDataFrame([], actual_df.schema).schema)

    # Sort by a unique key to ensure comparison is deterministic
    actual_sorted = sorted(actual_df.collect(), key=lambda r: r.id)
    expected_sorted = sorted(expected_df.collect(), key=lambda r: r.id)
    
    assert actual_sorted == expected_sorted


def test_add_age_and_eligibility_raises_error_if_dob_is_missing(spark):
    """
    Tests that a ValueError is raised if the input DataFrame does not
    contain a 'dob' column.
    """
    # Arrange
    source_data = [("Judy",)]
    source_schema = StructType([StructField("name", StringType())])
    source_df = spark.createDataFrame(source_data, source_schema)

    # Act & Assert
    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_eligibility_columns(source_df)