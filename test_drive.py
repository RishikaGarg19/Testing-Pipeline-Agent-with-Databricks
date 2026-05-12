import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
    IntegerType,
    BooleanType,
)
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# Import the function to be tested
from main import add_age_and_driving_status

# The SparkSession fixture MUST handle both Databricks (Spark Connect) and local environments.
# On Databricks, SPARK_REMOTE is set and .master() MUST NOT be called (it causes CANNOT_CONFIGURE_SPARK_CONNECT_MASTER).
# Locally, SPARK_REMOTE is absent and .master('local[1]') is needed.
# The spark fixture MUST look EXACTLY like this (copy it verbatim, do NOT modify it):
@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_add_age_and_driving_status_calculates_age_and_can_drive_correctly(spark):
    """
    Tests that age and driving status are correctly calculated for a standard set of inputs.
    """
    today = date.today()
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    input_data = [
        (1, today - relativedelta(years=25)),  # Adult who can drive
        (2, today - relativedelta(years=10)),  # Child who cannot drive
        (3, today - relativedelta(years=40, months=6)), # Adult, age should be floored
    ]
    input_df = spark.createDataFrame(input_data, schema)

    result_df = add_age_and_driving_status(input_df)

    expected_data = [
        Row(id=1, dob=today - relativedelta(years=25), age=25, canDrive=True),
        Row(id=2, dob=today - relativedelta(years=10), age=10, canDrive=False),
        Row(id=3, dob=today - relativedelta(years=40, months=6), age=40, canDrive=True),
    ]

    # Sort by ID to ensure deterministic comparison
    actual_data = sorted(result_df.collect(), key=lambda r: r.id)
    expected_data = sorted(expected_data, key=lambda r: r.id)

    assert actual_data == expected_data


def test_add_age_and_driving_status_handles_driving_age_boundaries(spark):
    """
    Tests the boundary conditions around the driving age of 16.
    """
    today = date.today()
    schema = StructType([
        StructField("description", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    input_data = [
        ("exactly_16", today - relativedelta(years=16)),
        ("almost_16", today - relativedelta(years=16) + timedelta(days=1)),
        ("just_over_16", today - relativedelta(years=16, days=1)),
    ]
    input_df = spark.createDataFrame(input_data, schema)

    result_df = add_age_and_driving_status(input_df)

    expected_data = {
        "exactly_16": Row(age=16, canDrive=True),
        "almost_16": Row(age=15, canDrive=False),
        "just_over_16": Row(age=16, canDrive=True),
    }

    result_data = {row["description"]: row for row in result_df.collect()}

    assert result_data["exactly_16"].age == expected_data["exactly_16"].age
    assert result_data["exactly_16"].canDrive == expected_data["exactly_16"].canDrive
    assert result_data["almost_16"].age == expected_data["almost_16"].age
    assert result_data["almost_16"].canDrive == expected_data["almost_16"].canDrive
    assert result_data["just_over_16"].age == expected_data["just_over_16"].age
    assert result_data["just_over_16"].canDrive == expected_data["just_over_16"].canDrive


def test_add_age_and_driving_status_handles_null_dob(spark):
    """
    Tests that a null value in the 'dob' column results in null 'age' and 'canDrive' as False.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    input_data = [(1, None)]
    input_df = spark.createDataFrame(input_data, schema)

    result_df = add_age_and_driving_status(input_df)

    # Note: When age is null, `age >= 16` evaluates to unknown, and `when` falls
    # through to the `otherwise(False)` clause.
    expected_data = [Row(id=1, dob=None, age=None, canDrive=False)]

    assert result_df.collect() == expected_data


def test_add_age_and_driving_status_preserves_duplicates(spark):
    """
    Tests that duplicate input rows are processed and remain duplicates in the output.
    """
    today = date.today()
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    dob = today - relativedelta(years=30)
    input_data = [
        (1, "Alice", dob),
        (2, "Bob", today - relativedelta(years=12)),
        (1, "Alice", dob),  # Duplicate
    ]
    input_df = spark.createDataFrame(input_data, schema)

    result_df = add_age_and_driving_status(input_df)

    assert result_df.count() == 3

    # Check that the duplicate row was processed correctly and is still a duplicate
    duplicate_rows = result_df.filter(F.col("id") == 1).collect()
    assert len(duplicate_rows) == 2
    expected_row = Row(id=1, name="Alice", dob=dob, age=30, canDrive=True)
    assert duplicate_rows[0] == expected_row
    assert duplicate_rows[1] == expected_row


def test_add_age_and_driving_status_handles_empty_dataframe(spark):
    """
    Tests that an empty DataFrame with the correct schema results in an empty DataFrame
    with the new columns added to its schema.
    """
    input_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    input_df = spark.createDataFrame([], input_schema)

    result_df = add_age_and_driving_status(input_df)

    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])

    assert result_df.count() == 0
    assert result_df.schema == expected_schema


def test_add_age_and_driving_status_raises_value_error_if_dob_is_missing(spark):
    """
    Tests that a ValueError is raised if the input DataFrame does not contain the 'dob' column.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
    ])
    input_data = [(1, "Charlie")]
    input_df = spark.createDataFrame(input_data, schema)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_driving_status(input_df)