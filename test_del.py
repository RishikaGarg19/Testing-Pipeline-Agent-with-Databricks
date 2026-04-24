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


@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_add_age_and_eligibility_columns_adult_can_vote(spark):
    """
    Tests that a person older than 18 is correctly identified as eligible to vote.
    """
    today = date.today()
    dob_adult = today - relativedelta(years=25)

    input_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    input_data = [("Alice", dob_adult)]
    input_df = spark.createDataFrame(input_data, schema=input_schema)

    actual_df = add_age_and_eligibility_columns(input_df)

    expected_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType()),
        StructField("age", IntegerType()),
        StructField("canVote", BooleanType())
    ])
    expected_data = [("Alice", dob_adult, 25, True)]
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

    assert actual_df.schema.fields == expected_df.schema.fields
    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_columns_minor_cannot_vote(spark):
    """
    Tests that a person younger than 18 is correctly identified as not eligible to vote.
    """
    today = date.today()
    dob_minor = today - relativedelta(years=16)

    input_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    input_data = [("Bob", dob_minor)]
    input_df = spark.createDataFrame(input_data, schema=input_schema)

    actual_df = add_age_and_eligibility_columns(input_df)

    expected_data = [Row(name="Bob", dob=dob_minor, age=16, canVote=False)]
    actual_data = actual_df.collect()

    assert len(actual_data) == 1
    assert actual_data == expected_data


def test_add_age_and_eligibility_columns_boundary_is_exactly_18(spark):
    """
    Tests the boundary condition where a person is exactly 18 years old today.
    """
    today = date.today()
    dob_18 = today - relativedelta(years=18)

    input_schema = StructType([StructField("dob", DateType())])
    input_data = [(dob_18,)]
    input_df = spark.createDataFrame(input_data, schema=input_schema)

    actual_df = add_age_and_eligibility_columns(input_df)

    expected_data = [Row(dob=dob_18, age=18, canVote=True)]
    actual_data = actual_df.collect()

    assert len(actual_data) == 1
    assert actual_data == expected_data


def test_add_age_and_eligibility_columns_boundary_day_before_18th_birthday(spark):
    """
    Tests the boundary condition where a person will turn 18 tomorrow.
    """
    today = date.today()
    dob_almost_18 = today - relativedelta(years=18) + relativedelta(days=1)

    input_schema = StructType([StructField("dob", DateType())])
    input_data = [(dob_almost_18,)]
    input_df = spark.createDataFrame(input_data, schema=input_schema)

    actual_df = add_age_and_eligibility_columns(input_df)

    expected_data = [Row(dob=dob_almost_18, age=17, canVote=False)]
    actual_data = actual_df.collect()

    assert len(actual_data) == 1
    assert actual_data == expected_data


def test_add_age_and_eligibility_columns_handles_null_dob(spark):
    """
    Tests that a null 'dob' results in a null 'age' and 'canVote' being False.
    """
    input_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    input_data = [("Charlie", None)]
    input_df = spark.createDataFrame(input_data, schema=input_schema)

    actual_df = add_age_and_eligibility_columns(input_df)

    expected_data = [Row(name="Charlie", dob=None, age=None, canVote=False)]
    actual_data = actual_df.collect()

    assert len(actual_data) == 1
    assert actual_data == expected_data


def test_add_age_and_eligibility_columns_preserves_other_columns(spark):
    """
    Tests that existing columns in the input DataFrame are preserved.
    """
    today = date.today()
    dob = today - relativedelta(years=30)

    input_schema = StructType([
        StructField("id", IntegerType()),
        StructField("dob", DateType()),
        StructField("city", StringType())
    ])
    input_data = [(101, dob, "New York")]
    input_df = spark.createDataFrame(input_data, schema=input_schema)

    actual_df = add_age_and_eligibility_columns(input_df)

    expected_schema = StructType([
        StructField("id", IntegerType()),
        StructField("dob", DateType()),
        StructField("city", StringType()),
        StructField("age", IntegerType()),
        StructField("canVote", BooleanType())
    ])
    expected_data = [(101, dob, "New York", 30, True)]
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

    assert actual_df.schema == expected_df.schema
    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_columns_handles_duplicates(spark):
    """
    Tests that duplicate input rows are processed correctly.
    """
    today = date.today()
    dob_adult = today - relativedelta(years=40)
    dob_minor = today - relativedelta(years=10)

    input_schema = StructType([StructField("dob", DateType())])
    input_data = [(dob_adult,), (dob_minor,), (dob_adult,)]
    input_df = spark.createDataFrame(input_data, schema=input_schema)

    actual_df = add_age_and_eligibility_columns(input_df)

    expected_data = [
        Row(dob=dob_adult, age=40, canVote=True),
        Row(dob=dob_minor, age=10, canVote=False),
        Row(dob=dob_adult, age=40, canVote=True)
    ]

    # Sorting to ensure comparison is deterministic
    actual_data_sorted = sorted(actual_df.collect(), key=lambda r: (r.dob, r.age))
    expected_data_sorted = sorted(expected_data, key=lambda r: (r.dob, r.age))

    assert actual_df.count() == 3
    assert actual_data_sorted == expected_data_sorted


def test_add_age_and_eligibility_columns_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the 'dob' column is not present.
    """
    input_schema = StructType([StructField("name", StringType())])
    input_data = [("David",)]
    input_df = spark.createDataFrame(input_data, schema=input_schema)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_eligibility_columns(input_df)


def test_add_age_and_eligibility_columns_with_mixed_data(spark):
    """
    Tests the function with a mix of adults, minors, and nulls.
    """
    today = date.today()
    dob_adult = today - relativedelta(years=50)
    dob_minor = today - relativedelta(years=15)
    dob_18 = today - relativedelta(years=18)

    input_schema = StructType([
        StructField("name", StringType()),
        StructField("dob", DateType())
    ])
    input_data = [
        ("Eve", dob_adult),
        ("Frank", dob_minor),
        ("Grace", None),
        ("Heidi", dob_18)
    ]
    input_df = spark.createDataFrame(input_data, schema=input_schema)

    actual_df = add_age_and_eligibility_columns(input_df)

    expected_data = [
        Row(name='Eve', dob=dob_adult, age=50, canVote=True),
        Row(name='Frank', dob=dob_minor, age=15, canVote=False),
        Row(name='Grace', dob=None, age=None, canVote=False),
        Row(name='Heidi', dob=dob_18, age=18, canVote=True)
    ]
    
    # Sorting to ensure comparison is deterministic
    actual_data_sorted = sorted(actual_df.collect(), key=lambda r: r.name)
    expected_data_sorted = sorted(expected_data, key=lambda r: r.name)
    
    assert actual_df.count() == 4
    assert actual_data_sorted == expected_data_sorted