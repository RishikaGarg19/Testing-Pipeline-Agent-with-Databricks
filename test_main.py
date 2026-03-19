import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
    IntegerType,
)
from datetime import date
from dateutil.relativedelta import relativedelta

from main import add_age_and_driving_status


@pytest.fixture(scope="session")
def spark():
    """
    Pytest fixture to create a SparkSession for the test suite.
    The session is stopped at the end of the tests.
    """
    session = (
        SparkSession.builder.master("local[2]")
        .appName("ETLPipelineTests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.memory", "512m")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_add_age_and_driving_status_calculates_age_and_status_correctly(spark):
    """
    Tests the basic calculation for people over and under the driving age.
    """
    today = date.today()
    dob_over_18 = today - relativedelta(years=25)
    dob_under_18 = today - relativedelta(years=10)

    source_data = [
        Row(id=1, name="Alex", dob=dob_over_18),
        Row(id=2, name="Beth", dob=dob_under_18),
    ]
    source_df = spark.createDataFrame(source_data)

    actual_df = add_age_and_driving_status(source_df)

    expected_data = [
        Row(id=1, name="Alex", dob=dob_over_18, age=25, canDrive="Yes"),
        Row(id=2, name="Beth", dob=dob_under_18, age=10, canDrive="No"),
    ]
    expected_df = spark.createDataFrame(expected_data)

    actual_data = sorted(actual_df.collect(), key=lambda r: r.id)
    expected_data_sorted = sorted(expected_df.collect(), key=lambda r: r.id)

    assert actual_data == expected_data_sorted
    assert "age" in actual_df.columns
    assert "canDrive" in actual_df.columns


def test_add_age_and_driving_status_handles_null_dob(spark):
    """
    Tests that a null 'dob' results in a null 'age' and 'No' for 'canDrive'.
    """
    source_data = [Row(id=1, name="Charlie", dob=None)]
    source_df = spark.createDataFrame(source_data)

    actual_df = add_age_and_driving_status(source_df)

    expected_data = [Row(id=1, name="Charlie", dob=None, age=None, canDrive="No")]
    # Define schema explicitly to check nullability
    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", StringType(), False)
    ])
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

    assert actual_df.collect() == expected_df.collect()
    assert actual_df.schema == expected_df.schema


def test_add_age_and_driving_status_boundary_exactly_18_years_old(spark):
    """
    Tests the boundary case where a person's 18th birthday is today.
    """
    today = date.today()
    dob_exactly_18 = today - relativedelta(years=18)

    source_data = [Row(id=1, name="Dana", dob=dob_exactly_18)]
    source_df = spark.createDataFrame(source_data)

    actual_df = add_age_and_driving_status(source_df)

    expected_data = [Row(id=1, name="Dana", dob=dob_exactly_18, age=18, canDrive="Yes")]
    expected_df = spark.createDataFrame(expected_data)

    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_driving_status_boundary_one_day_before_18th_birthday(spark):
    """
    Tests the boundary case where a person will be 18 tomorrow.
    """
    today = date.today()
    dob_almost_18 = today - relativedelta(years=18) + relativedelta(days=1)

    source_data = [Row(id=1, name="Eli", dob=dob_almost_18)]
    source_df = spark.createDataFrame(source_data)

    actual_df = add_age_and_driving_status(source_df)

    expected_data = [Row(id=1, name="Eli", dob=dob_almost_18, age=17, canDrive="No")]
    expected_df = spark.createDataFrame(expected_data)

    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_driving_status_handles_duplicate_rows(spark):
    """
    Tests that duplicate input rows are processed correctly and result in
    duplicate output rows.
    """
    today = date.today()
    dob = today - relativedelta(years=22)
    source_data = [
        Row(id=1, name="Fran", dob=dob),
        Row(id=1, name="Fran", dob=dob),
    ]
    source_df = spark.createDataFrame(source_data)

    actual_df = add_age_and_driving_status(source_df)

    expected_data = [
        Row(id=1, name="Fran", dob=dob, age=22, canDrive="Yes"),
        Row(id=1, name="Fran", dob=dob, age=22, canDrive="Yes"),
    ]
    expected_df = spark.createDataFrame(expected_data)

    assert actual_df.count() == 2
    assert actual_df.collect() == expected_df.collect()


def test_add_age_and_driving_status_with_empty_dataframe(spark):
    """
    Tests that an empty DataFrame with the correct schema results in an empty
    DataFrame with the new columns added.
    """
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame([], schema=source_schema)

    actual_df = add_age_and_driving_status(source_df)

    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", StringType(), False),
    ])

    assert actual_df.count() == 0
    assert actual_df.schema == expected_schema


def test_add_age_and_driving_status_raises_error_if_dob_column_is_missing(spark):
    """
    Tests that a ValueError is raised if the input DataFrame does not
    contain the required 'dob' column.
    """
    source_data = [Row(id=1, name="George")]
    source_df = spark.createDataFrame(source_data)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_driving_status(source_df)