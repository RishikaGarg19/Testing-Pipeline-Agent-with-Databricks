import pytest
from pyspark.sql import SparkSession, Row
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
from main import add_age_and_eligibility_columns


@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_add_age_and_eligibility_columns_for_adult(spark):
    """
    Tests that a person over 18 is correctly identified as eligible to drive and vote.
    """
    today = date.today()
    dob_adult = today - relativedelta(years=25)

    source_data = [Row(id=1, name="John Doe", dob=dob_adult)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [
        Row(id=1, name="John Doe", dob=dob_adult, age=25, canDrive=True, canVote=True)
    ]
    expected_df = spark.createDataFrame(expected_data)

    actual_df = add_age_and_eligibility_columns(source_df)

    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_columns_for_minor(spark):
    """
    Tests that a person under 18 is correctly identified as not eligible.
    """
    today = date.today()
    dob_minor = today - relativedelta(years=16)

    source_data = [Row(id=2, name="Jane Smith", dob=dob_minor)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [
        Row(id=2, name="Jane Smith", dob=dob_minor, age=16, canDrive=False, canVote=False)
    ]
    expected_df = spark.createDataFrame(expected_data)

    actual_df = add_age_and_eligibility_columns(source_df)

    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_columns_boundary_is_18(spark):
    """
    Tests the boundary case where a person's 18th birthday is today.
    """
    today = date.today()
    dob_18_today = today - relativedelta(years=18)

    source_data = [Row(id=3, name="Eighteen Today", dob=dob_18_today)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [
        Row(
            id=3,
            name="Eighteen Today",
            dob=dob_18_today,
            age=18,
            canDrive=True,
            canVote=True,
        )
    ]
    expected_df = spark.createDataFrame(expected_data)

    actual_df = add_age_and_eligibility_columns(source_df)

    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_columns_boundary_under_18(spark):
    """
    Tests the boundary case where a person's 18th birthday is tomorrow.
    """
    today = date.today()
    dob_almost_18 = today - relativedelta(years=18) + relativedelta(days=1)

    source_data = [Row(id=4, name="Almost Eighteen", dob=dob_almost_18)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [
        Row(
            id=4,
            name="Almost Eighteen",
            dob=dob_almost_18,
            age=17,
            canDrive=False,
            canVote=False,
        )
    ]
    expected_df = spark.createDataFrame(expected_data)

    actual_df = add_age_and_eligibility_columns(source_df)

    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_columns_with_null_dob(spark):
    """
    Tests that a null 'dob' results in a null 'age' and False for eligibility columns.
    """
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_data = [(1, "No DOB", None)]
    source_df = spark.createDataFrame(source_data, schema=source_schema)

    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])
    expected_data = [(1, "No DOB", None, None, False, False)]
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

    actual_df = add_age_and_eligibility_columns(source_df)

    # Comparing schemas separately for clarity
    assert actual_df.schema == expected_df.schema
    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_eligibility_columns_handles_duplicates(spark):
    """
    Tests that duplicate input rows are processed and remain duplicates in the output.
    """
    today = date.today()
    dob = today - relativedelta(years=40)

    source_data = [
        Row(id=5, name="Duplicate Person", dob=dob),
        Row(id=5, name="Duplicate Person", dob=dob),
    ]
    source_df = spark.createDataFrame(source_data)

    expected_data = [
        Row(id=5, name="Duplicate Person", dob=dob, age=40, canDrive=True, canVote=True),
        Row(id=5, name="Duplicate Person", dob=dob, age=40, canDrive=True, canVote=True),
    ]
    expected_df = spark.createDataFrame(expected_data)

    actual_df = add_age_and_eligibility_columns(source_df)

    assert sorted(actual_df.collect()) == sorted(expected_df.collect())


def test_add_age_and_eligibility_columns_has_correct_schema(spark):
    """
    Tests that the output DataFrame has the expected schema and data types.
    """
    source_data = [Row(id=1, dob=date(2000, 1, 1))]
    source_df = spark.createDataFrame(source_data)

    actual_df = add_age_and_eligibility_columns(source_df)
    actual_schema = actual_df.schema

    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])

    assert actual_schema == expected_schema


def test_add_age_and_eligibility_columns_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the 'dob' column is not in the input DataFrame.
    """
    source_data = [Row(id=1, name="No DOB Column")]
    source_df = spark.createDataFrame(source_data)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_eligibility_columns(source_df)


def test_add_age_and_eligibility_columns_with_empty_dataframe(spark):
    """
    Tests that the function handles an empty DataFrame correctly.
    """
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame([], schema=source_schema)

    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False),
        StructField("canVote", BooleanType(), False),
    ])
    expected_df = spark.createDataFrame([], schema=expected_schema)

    actual_df = add_age_and_eligibility_columns(source_df)

    assert actual_df.schema == expected_df.schema
    assert actual_df.collect() == expected_df.collect()