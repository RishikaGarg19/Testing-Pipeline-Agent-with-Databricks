import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DateType,
    BooleanType,
)
from datetime import date
from dateutil.relativedelta import relativedelta

# Import the function to be tested
from main import add_age_and_driving_status


@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_add_age_and_driving_status_can_drive_true(spark):
    """
    Tests that a person older than 16 is correctly identified as being able to drive.
    """
    # Dynamic date of birth for someone who is 25 years old
    dob_can_drive = date.today() - relativedelta(years=25)

    source_data = [
        (1, "Alice", dob_can_drive)
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    expected_data = [
        (1, "Alice", dob_can_drive, 25, True)
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    result_df = add_age_and_driving_status(source_df)

    # Comparing dataframes by collecting and sorting
    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema == expected_df.schema


def test_add_age_and_driving_status_can_drive_false(spark):
    """
    Tests that a person younger than 16 is correctly identified as not being able to drive.
    """
    # Dynamic date of birth for someone who is 15 years old
    dob_cannot_drive = date.today() - relativedelta(years=15)

    source_data = [
        (2, "Bob", dob_cannot_drive)
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    expected_data = [
        (2, "Bob", dob_cannot_drive, 15, False)
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    result_df = add_age_and_driving_status(source_df)

    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema == expected_df.schema


def test_add_age_and_driving_status_boundary_exactly_16(spark):
    """
    Tests the boundary condition where a person is exactly 16 years old.
    """
    # Dynamic date of birth for someone who turned 16 today
    dob_exactly_16 = date.today() - relativedelta(years=16)

    source_data = [
        (3, "Charlie", dob_exactly_16)
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    expected_data = [
        (3, "Charlie", dob_exactly_16, 16, True)
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    result_df = add_age_and_driving_status(source_df)

    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema == expected_df.schema


def test_add_age_and_driving_status_boundary_almost_16(spark):
    """
    Tests the boundary condition where a person is one day shy of their 16th birthday.
    """
    # Dynamic date of birth for someone who will turn 16 tomorrow
    dob_almost_16 = (date.today() - relativedelta(years=16)) + relativedelta(days=1)

    source_data = [
        (4, "Diana", dob_almost_16)
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    expected_data = [
        (4, "Diana", dob_almost_16, 15, False)
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    result_df = add_age_and_driving_status(source_df)

    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema == expected_df.schema


def test_add_age_and_driving_status_with_null_dob(spark):
    """
    Tests that a row with a null 'dob' results in a null 'age' and 'canDrive' as False.
    """
    source_data = [
        (5, "Eve", None)
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    expected_data = [
        (5, "Eve", None, None, False)
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    result_df = add_age_and_driving_status(source_df)

    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema == expected_df.schema


def test_add_age_and_driving_status_with_duplicates(spark):
    """
    Tests that duplicate input rows are processed correctly and result in duplicate output rows.
    """
    dob = date.today() - relativedelta(years=30)
    source_data = [
        (6, "Frank", dob),
        (6, "Frank", dob)
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    expected_data = [
        (6, "Frank", dob, 30, True),
        (6, "Frank", dob, 30, True)
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    result_df = add_age_and_driving_status(source_df)

    assert result_df.count() == 2
    assert sorted(result_df.collect()) == sorted(expected_df.collect())


def test_add_age_and_driving_status_empty_dataframe(spark):
    """
    Tests that the function handles an empty DataFrame correctly, returning an empty DataFrame
    with the new columns in the schema.
    """
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame([], source_schema)

    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])

    result_df = add_age_and_driving_status(source_df)

    assert result_df.count() == 0
    assert result_df.schema == expected_schema


def test_add_age_and_driving_status_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the 'dob' column is not in the input DataFrame.
    """
    source_data = [
        (7, "Grace")
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_driving_status(source_df)