import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DateType,
    BooleanType,
    LongType
)
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# Import functions from the ETL script
from main import add_age_and_voting_status

# This fixture is required for all PySpark tests.
# It is designed to work both in local development and on Databricks.
# DO NOT MODIFY THIS FIXTURE.
@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_add_age_and_voting_status_calculates_age_and_can_vote_correctly(spark):
    """
    Tests that the function correctly calculates age and voting status for various dates of birth.
    """
    today = date.today()
    test_data = [
        (1, "Alice", today - relativedelta(years=30)),  # Well over 18
        (2, "Bob", today - relativedelta(years=17, months=11)),  # Just under 18
        (3, "Charlie", today - relativedelta(years=65)),  # Senior citizen
        (4, "Diana", today - relativedelta(years=1)),  # A baby
    ]
    input_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    input_df = spark.createDataFrame(data=test_data, schema=input_schema)

    result_df = add_age_and_voting_status(input_df)

    expected_data = [
        (1, "Alice", today - relativedelta(years=30), 30, True),
        (2, "Bob", today - relativedelta(years=17, months=11), 17, False),
        (3, "Charlie", today - relativedelta(years=65), 65, True),
        (4, "Diana", today - relativedelta(years=1), 1, False),
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    # Comparing data as sets of dictionaries to ignore row order
    assert result_df.schema == expected_df.schema
    assert [row.asDict() for row in result_df.collect()] == [row.asDict() for row in expected_df.collect()]


def test_add_age_and_voting_status_handles_18th_birthday_boundary(spark):
    """
    Tests the boundary cases for turning 18: today is the 18th birthday and tomorrow is the 18th birthday.
    """
    today = date.today()
    test_data = [
        (1, "Eve", today - relativedelta(years=18)),  # Exactly 18 today
        (2, "Frank", today - relativedelta(years=18) + timedelta(days=1)),  # Turns 18 tomorrow
    ]
    input_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    input_df = spark.createDataFrame(data=test_data, schema=input_schema)

    result_df = add_age_and_voting_status(input_df)

    expected_data = [
        (1, "Eve", today - relativedelta(years=18), 18, True),
        (2, "Frank", today - relativedelta(years=18) + timedelta(days=1), 17, False),
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    assert result_df.schema == expected_df.schema
    assert [row.asDict() for row in result_df.collect()] == [row.asDict() for row in expected_df.collect()]


def test_add_age_and_voting_status_handles_null_dob(spark):
    """
    Tests that a null 'dob' results in a null 'age' and a 'canVote' status of False.
    """
    test_data = [(1, "Grace", None)]
    input_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    input_df = spark.createDataFrame(data=test_data, schema=input_schema)

    result_df = add_age_and_voting_status(input_df)

    expected_data = [(1, "Grace", None, None, False)]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    assert result_df.schema == expected_df.schema
    assert [row.asDict() for row in result_df.collect()] == [row.asDict() for row in expected_df.collect()]


def test_add_age_and_voting_status_handles_duplicate_rows(spark):
    """
    Tests that duplicate input rows are processed correctly without being dropped.
    """
    today = date.today()
    dob_over_18 = today - relativedelta(years=25)
    test_data = [
        (1, "Heidi", dob_over_18),
        (1, "Heidi", dob_over_18),  # Duplicate row
    ]
    input_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    input_df = spark.createDataFrame(data=test_data, schema=input_schema)

    result_df = add_age_and_voting_status(input_df)

    expected_data = [
        (1, "Heidi", dob_over_18, 25, True),
        (1, "Heidi", dob_over_18, 25, True),
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    assert result_df.count() == 2
    assert [row.asDict() for row in result_df.collect()] == [row.asDict() for row in expected_df.collect()]


def test_add_age_and_voting_status_handles_empty_dataframe(spark):
    """
    Tests that the function returns an empty DataFrame with the correct schema when given an empty DataFrame.
    """
    input_schema = StructType([
        StructField("id", LongType(), True),
        StructField("dob", DateType(), True),
    ])
    input_df = spark.createDataFrame([], schema=input_schema)

    result_df = add_age_and_voting_status(input_df)

    expected_schema = StructType([
        StructField("id", LongType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), True),
    ])

    assert result_df.count() == 0
    assert result_df.schema == expected_schema


def test_add_age_and_voting_status_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the input DataFrame does not contain a 'dob' column.
    """
    test_data = [(1, "Ivan")]
    input_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
    ])
    input_df = spark.createDataFrame(data=test_data, schema=input_schema)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_voting_status(input_df)