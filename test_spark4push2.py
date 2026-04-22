import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, BooleanType
from datetime import date
from dateutil.relativedelta import relativedelta

from main import add_age_and_driving_status


@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_add_age_and_driving_status_can_drive(spark):
    """
    Tests that a person older than 16 is correctly identified as being able to drive.
    """
    today = date.today()
    dob_18_years_ago = today - relativedelta(years=18)
    
    source_data = [
        {"id": 1, "dob": dob_18_years_ago},
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    result_df = add_age_and_driving_status(source_df)
    
    expected_data = [
        {"id": 1, "dob": dob_18_years_ago, "age": 18, "canDrive": True},
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert result_df.collect() == expected_df.collect()
    assert result_df.schema == expected_df.schema


def test_add_age_and_driving_status_cannot_drive(spark):
    """
    Tests that a person younger than 16 is correctly identified as not being able to drive.
    """
    today = date.today()
    dob_15_years_ago = today - relativedelta(years=15)
    
    source_data = [
        {"id": 1, "dob": dob_15_years_ago},
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    result_df = add_age_and_driving_status(source_df)
    
    expected_data = [
        {"id": 1, "dob": dob_15_years_ago, "age": 15, "canDrive": False},
    ]
    expected_df = spark.createDataFrame(expected_data, result_df.schema)
    
    assert result_df.collect() == expected_df.collect()


def test_add_age_and_driving_status_exactly_16_birthday_today(spark):
    """
    Tests the boundary condition where a person turns exactly 16 today.
    """
    today = date.today()
    dob_16_years_ago = today - relativedelta(years=16)
    
    source_data = [
        {"id": 1, "dob": dob_16_years_ago},
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    result_df = add_age_and_driving_status(source_df)
    
    expected_data = [
        {"id": 1, "dob": dob_16_years_ago, "age": 16, "canDrive": True},
    ]
    expected_df = spark.createDataFrame(expected_data, result_df.schema)
    
    assert result_df.collect() == expected_df.collect()


def test_add_age_and_driving_status_day_before_16th_birthday(spark):
    """
    Tests the boundary condition where a person will turn 16 tomorrow.
    """
    today = date.today()
    dob_turning_16_tomorrow = today - relativedelta(years=16) + relativedelta(days=1)
    
    source_data = [
        {"id": 1, "dob": dob_turning_16_tomorrow},
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    result_df = add_age_and_driving_status(source_df)
    
    expected_data = [
        {"id": 1, "dob": dob_turning_16_tomorrow, "age": 15, "canDrive": False},
    ]
    expected_df = spark.createDataFrame(expected_data, result_df.schema)
    
    assert result_df.collect() == expected_df.collect()


def test_add_age_and_driving_status_with_null_dob(spark):
    """
    Tests the edge case where the 'dob' is null. Age should be null and canDrive should be false.
    """
    source_data = [
        {"id": 1, "dob": None},
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    result_df = add_age_and_driving_status(source_df)
    
    expected_data = [
        {"id": 1, "dob": None, "age": None, "canDrive": False},
    ]
    expected_df = spark.createDataFrame(expected_data, result_df.schema)
    
    assert result_df.collect() == expected_df.collect()


def test_add_age_and_driving_status_with_duplicates(spark):
    """
    Tests that duplicate input rows are processed correctly and preserved.
    """
    today = date.today()
    dob_20_years_ago = today - relativedelta(years=20)
    
    source_data = [
        {"id": 1, "dob": dob_20_years_ago},
        {"id": 1, "dob": dob_20_years_ago},
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    result_df = add_age_and_driving_status(source_df)
    
    expected_data = [
        {"id": 1, "dob": dob_20_years_ago, "age": 20, "canDrive": True},
        {"id": 1, "dob": dob_20_years_ago, "age": 20, "canDrive": True},
    ]
    expected_df = spark.createDataFrame(expected_data, result_df.schema)
    
    assert result_df.count() == 2
    assert result_df.collect() == expected_df.collect()


def test_add_age_and_driving_status_empty_dataframe(spark):
    """
    Tests that an empty DataFrame is handled correctly, returning an empty DataFrame with the new columns.
    """
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame([], source_schema)
    
    result_df = add_age_and_driving_status(source_df)
    
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
    ])
    
    assert result_df.count() == 0
    assert result_df.schema == expected_schema


def test_add_age_and_driving_status_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the 'dob' column is missing from the input DataFrame.
    """
    source_data = [
        {"id": 1, "name": "John Doe"},
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_driving_status(source_df)