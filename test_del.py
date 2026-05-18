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

from main import add_age_and_voting_status


@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_add_age_and_voting_status_for_adult(spark):
    """
    Tests that a person older than 18 is correctly identified as being able to vote.
    """
    today = date.today()
    dob_adult = today - relativedelta(years=25)

    source_data = [(1, "John Doe", dob_adult)]
    source_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    result_df = add_age_and_voting_status(source_df)

    expected_data = [(1, "John Doe", dob_adult, 25, True)]
    expected_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema.simpleString() == expected_df.schema.simpleString()


def test_add_age_and_voting_status_for_minor(spark):
    """
    Tests that a person younger than 18 is correctly identified as not being able to vote.
    """
    today = date.today()
    dob_minor = today - relativedelta(years=15)

    source_data = [(2, "Jane Smith", dob_minor)]
    source_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    result_df = add_age_and_voting_status(source_df)

    expected_data = [(2, "Jane Smith", dob_minor, 15, False)]
    # 'canVote' schema is not nullable based on the function's logic
    expected_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assert result_df.count() == 1
    # Adjust schema of expected DF to match actual for comparison
    result_df = result_df.withColumn("canVote", result_df["canVote"].cast(BooleanType()))
    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema["canVote"].nullable is False


def test_add_age_and_voting_status_boundary_is_exactly_18(spark):
    """
    Tests the boundary condition where a person's 18th birthday is today.
    """
    today = date.today()
    dob_18th_bday = today - relativedelta(years=18)

    source_data = [(3, "Emily Brown", dob_18th_bday)]
    source_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    result_df = add_age_and_voting_status(source_df)

    expected_data = [(3, "Emily Brown", dob_18th_bday, 18, True)]
    expected_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())


def test_add_age_and_voting_status_boundary_day_before_18th(spark):
    """
    Tests the boundary condition where a person's 18th birthday is tomorrow.
    """
    today = date.today()
    # If today is 2023-10-27, 18 years ago was 2005-10-27.
    # To be 17, their birthday must be 2005-10-28.
    dob_almost_18 = today - relativedelta(years=18) + relativedelta(days=1)

    source_data = [(4, "Michael Lee", dob_almost_18)]
    source_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    result_df = add_age_and_voting_status(source_df)

    expected_data = [(4, "Michael Lee", dob_almost_18, 17, False)]
    expected_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assert result_df.count() == 1
    # Adjust schema of expected DF to match actual for comparison
    result_df = result_df.withColumn("canVote", result_df["canVote"].cast(BooleanType()))
    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema["canVote"].nullable is False


def test_add_age_and_voting_status_with_null_dob(spark):
    """
    Tests that a record with a null 'dob' results in a null 'age' and 'canVote' as False.
    """
    source_data = [(5, "Chris Green", None)]
    source_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    result_df = add_age_and_voting_status(source_df)

    expected_data = [(5, "Chris Green", None, None, False)]
    expected_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assert result_df.count() == 1
    # Adjust schema of expected DF to match actual for comparison
    result_df = result_df.withColumn("canVote", result_df["canVote"].cast(BooleanType()))
    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema["canVote"].nullable is False


def test_add_age_and_voting_status_with_duplicate_records(spark):
    """
    Tests that duplicate input records are processed correctly and result in duplicate output records.
    """
    today = date.today()
    dob_adult = today - relativedelta(years=40)
    dob_minor = today - relativedelta(years=10)

    source_data = [
        (6, "Pat Ray", dob_adult),
        (7, "Alex Kim", dob_minor),
        (6, "Pat Ray", dob_adult),  # Duplicate
    ]
    source_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    result_df = add_age_and_voting_status(source_df)

    expected_data = [
        (6, "Pat Ray", dob_adult, 40, True),
        (7, "Alex Kim", dob_minor, 10, False),
        (6, "Pat Ray", dob_adult, 40, True),
    ]
    expected_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assert result_df.count() == 3
    assert sorted(result_df.collect()) == sorted(expected_df.collect())


def test_add_age_and_voting_status_with_empty_dataframe(spark):
    """
    Tests that the function handles an empty DataFrame correctly, returning an empty DataFrame with the new columns.
    """
    source_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
    ])
    source_df = spark.createDataFrame([], source_schema)

    result_df = add_age_and_voting_status(source_df)

    expected_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("dob", DateType(), False),
        StructField("age", IntegerType(), True),
        StructField("canVote", BooleanType(), False),
    ])

    assert result_df.count() == 0
    # canVote is not-nullable because of the otherwise(False) clause
    assert result_df.schema.simpleString() == expected_schema.simpleString()


def test_add_age_and_voting_status_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the input DataFrame does not contain the 'dob' column.
    """
    source_data = [(1, "No Dob")]
    source_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), False),
    ])
    source_df = spark.createDataFrame(source_data, source_schema)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_voting_status(source_df)