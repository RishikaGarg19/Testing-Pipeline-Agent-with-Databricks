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
import collections

from main import transform_customers_data

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


def assert_df_equality(actual_df, expected_df):
    """
    Helper function to assert two DataFrames are equal, ignoring row order.
    """
    assert actual_df.schema == expected_df.schema, "Schemas do not match"
    
    actual_data = sorted(actual_df.collect(), key=lambda r: str(r))
    expected_data = sorted(expected_df.collect(), key=lambda r: str(r))
    
    assert actual_data == expected_data, "Data does not match"


def test_transform_customers_data_calculates_age_can_drive_and_can_vote(spark):
    """
    Tests that age, canDrive, and canVote are calculated correctly for an adult over 20.
    """
    today = date.today()
    dob_30_years_ago = today - relativedelta(years=30)
    
    source_data = [("Alice", dob_30_years_ago)]
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    actual_df = transform_customers_data(source_df)
    
    expected_data = [("Alice", dob_30_years_ago, 30, True, True)]
    expected_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert_df_equality(actual_df, expected_df)


def test_transform_customers_data_boundary_age_18(spark):
    """
    Tests the boundary condition for a person who is exactly 18.
    They should be able to drive but not vote.
    """
    today = date.today()
    dob_18_years_ago = today - relativedelta(years=18)
    
    source_data = [("Bob", dob_18_years_ago)]
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    actual_df = transform_customers_data(source_df)

    expected_data = [("Bob", dob_18_years_ago, 18, True, False)]
    expected_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert_df_equality(actual_df, expected_df)


def test_transform_customers_data_boundary_age_under_18(spark):
    """
    Tests the boundary condition for a person who is 17.
    They should not be able to drive or vote.
    """
    today = date.today()
    dob_17_years_ago = today - relativedelta(years=17)
    
    source_data = [("Charlie", dob_17_years_ago)]
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    actual_df = transform_customers_data(source_df)

    expected_data = [("Charlie", dob_17_years_ago, 17, False, False)]
    expected_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert_df_equality(actual_df, expected_df)


def test_transform_customers_data_boundary_age_20(spark):
    """
    Tests the boundary condition for a person who is exactly 20.
    They should be able to drive and vote.
    """
    today = date.today()
    dob_20_years_ago = today - relativedelta(years=20)
    
    source_data = [("Diana", dob_20_years_ago)]
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    actual_df = transform_customers_data(source_df)

    expected_data = [("Diana", dob_20_years_ago, 20, True, True)]
    expected_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert_df_equality(actual_df, expected_df)
    

def test_transform_customers_data_boundary_age_19(spark):
    """
    Tests the boundary condition for a person who is 19.
    They should be able to drive but not vote.
    """
    today = date.today()
    dob_19_years_ago = today - relativedelta(years=19)
    
    source_data = [("Eve", dob_19_years_ago)]
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    actual_df = transform_customers_data(source_df)

    expected_data = [("Eve", dob_19_years_ago, 19, True, False)]
    expected_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert_df_equality(actual_df, expected_df)


def test_transform_customers_data_handles_null_dob(spark):
    """
    Tests that a null 'dob' results in null 'age' and false for 'canDrive' and 'canVote'.
    """
    source_data = [("Frank", None)]
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    actual_df = transform_customers_data(source_df)
    
    expected_data = [("Frank", None, None, False, False)]
    expected_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert_df_equality(actual_df, expected_df)


def test_transform_customers_data_handles_duplicate_rows(spark):
    """
    Tests that duplicate input rows are processed correctly and result in duplicate output rows.
    """
    today = date.today()
    dob = today - relativedelta(years=25)
    
    source_data = [
        ("Grace", dob),
        ("Grace", dob)
    ]
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    actual_df = transform_customers_data(source_df)
    
    expected_data = [
        ("Grace", dob, 25, True, True),
        ("Grace", dob, 25, True, True)
    ]
    expected_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)
    
    assert actual_df.count() == 2
    assert_df_equality(actual_df, expected_df)


def test_transform_customers_data_raises_value_error_if_dob_is_missing(spark):
    """
    Tests that a ValueError is raised if the input DataFrame does not contain a 'dob' column.
    """
    source_data = [("Heidi",)]
    source_schema = StructType([StructField("name", StringType(), True)])
    source_df = spark.createDataFrame(source_data, source_schema)
    
    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        transform_customers_data(source_df)