import pytest
from pyspark.sql import SparkSession, Row, DataFrame
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
import os

from main import add_age_and_driving_status


@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def assert_df_equality(df1: DataFrame, df2: DataFrame):
    """Asserts that two DataFrames are equal, ignoring column and row order."""
    assert sorted(df1.columns) == sorted(df2.columns), "Column names do not match"
    
    # Canonicalize by sorting columns and rows
    df1_sorted = df1.select(sorted(df1.columns))
    df2_sorted = df2.select(sorted(df2.columns))
    
    rows1 = sorted(df1_sorted.collect())
    rows2 = sorted(df2_sorted.collect())
    
    assert rows1 == rows2, "DataFrame content does not match"


def test_add_age_and_driving_status_adult_can_drive(spark):
    """Test with an adult well over 16, who should be able to drive."""
    today = date.today()
    dob = today - relativedelta(years=30)
    
    input_data = [Row(id=1, name="Alice", dob=dob)]
    input_df = spark.createDataFrame(input_data)

    result_df = add_age_and_driving_status(input_df)

    expected_data = [Row(id=1, name="Alice", dob=dob, age=30, canDrive=True)]
    expected_df = spark.createDataFrame(expected_data)

    assert_df_equality(result_df, expected_df)


def test_add_age_and_driving_status_child_cannot_drive(spark):
    """Test with a child under 16, who should not be able to drive."""
    today = date.today()
    dob = today - relativedelta(years=10)
    
    input_data = [Row(id=2, name="Bob", dob=dob)]
    input_df = spark.createDataFrame(input_data)

    result_df = add_age_and_driving_status(input_df)

    expected_data = [Row(id=2, name="Bob", dob=dob, age=10, canDrive=False)]
    expected_df = spark.createDataFrame(expected_data)

    assert_df_equality(result_df, expected_df)


def test_add_age_and_driving_status_boundary_just_turned_16(spark):
    """Test the boundary case of a person whose 16th birthday is today."""
    today = date.today()
    dob = today - relativedelta(years=16)

    input_data = [Row(id=3, name="Charlie", dob=dob)]
    input_df = spark.createDataFrame(input_data)

    result_df = add_age_and_driving_status(input_df)

    expected_data = [Row(id=3, name="Charlie", dob=dob, age=16, canDrive=True)]
    expected_df = spark.createDataFrame(expected_data)

    assert_df_equality(result_df, expected_df)


def test_add_age_and_driving_status_boundary_turning_16_tomorrow(spark):
    """Test the boundary case of a person who is 15 and turns 16 tomorrow."""
    today = date.today()
    dob = today - relativedelta(years=16) + timedelta(days=1)

    input_data = [Row(id=4, name="Dana", dob=dob)]
    input_df = spark.createDataFrame(input_data)

    result_df = add_age_and_driving_status(input_df)

    expected_data = [Row(id=4, name="Dana", dob=dob, age=15, canDrive=False)]
    expected_df = spark.createDataFrame(expected_data)

    assert_df_equality(result_df, expected_df)


def test_add_age_and_driving_status_null_dob(spark):
    """Test with a null 'dob', expecting a null 'age' and 'canDrive' as False."""
    input_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True)
    ])
    input_data = [(1, None)]
    input_df = spark.createDataFrame(input_data, schema=input_schema)
    
    result_df = add_age_and_driving_status(input_df)
    
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False)  # Non-nullable due to otherwise(lit(False))
    ])
    expected_data = [(1, None, None, False)]
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

    assert result_df.schema == expected_df.schema
    assert result_df.collect() == expected_df.collect()


def test_add_age_and_driving_status_multiple_rows_and_types(spark):
    """Test with multiple rows including various cases: adult, child, null, and boundary."""
    today = date.today()
    dob_adult = today - relativedelta(years=25)
    dob_child = today - relativedelta(years=10)
    dob_boundary = today - relativedelta(years=16) + timedelta(days=1)
    
    input_data = [
        Row(name="Alice", dob=dob_adult),
        Row(name="Bob", dob=dob_child),
        Row(name="Charlie", dob=dob_boundary),
        Row(name="David", dob=None)
    ]
    input_df = spark.createDataFrame(input_data)
    
    result_df = add_age_and_driving_status(input_df)
    
    expected_data = [
        Row(name="Alice", dob=dob_adult, age=25, canDrive=True),
        Row(name="Bob", dob=dob_child, age=10, canDrive=False),
        Row(name="Charlie", dob=dob_boundary, age=15, canDrive=False),
        Row(name="David", dob=None, age=None, canDrive=False)
    ]
    expected_df = spark.createDataFrame(expected_data)
    
    assert_df_equality(result_df, expected_df)


def test_add_age_and_driving_status_duplicate_rows(spark):
    """Test with duplicate input rows to ensure they are processed correctly and not de-duplicated."""
    today = date.today()
    dob = today - relativedelta(years=20)
    
    input_data = [
        Row(id=1, dob=dob),
        Row(id=1, dob=dob)
    ]
    input_df = spark.createDataFrame(input_data)
    
    result_df = add_age_and_driving_status(input_df)
    
    expected_data = [
        Row(id=1, dob=dob, age=20, canDrive=True),
        Row(id=1, dob=dob, age=20, canDrive=True)
    ]
    expected_df = spark.createDataFrame(expected_data)
    
    assert result_df.count() == 2
    assert_df_equality(result_df, expected_df)


def test_add_age_and_driving_status_empty_dataframe(spark):
    """Test that the function handles an empty DataFrame gracefully, preserving the schema."""
    input_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True)
    ])
    input_df = spark.createDataFrame([], schema=input_schema)
    
    result_df = add_age_and_driving_status(input_df)
    
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False)
    ])
    
    assert result_df.schema == expected_schema
    assert result_df.count() == 0


def test_add_age_and_driving_status_missing_dob_column_raises_error(spark):
    """Test that a ValueError is raised if the 'dob' column is missing."""
    input_data = [Row(id=1, name='test')]
    input_df = spark.createDataFrame(input_data)
    
    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_driving_status(input_df)