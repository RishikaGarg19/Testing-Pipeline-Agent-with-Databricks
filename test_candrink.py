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
from unittest.mock import patch

# Import functions from the main script
from main import add_age_and_drinking_status, main


@pytest.fixture(scope="session")
def spark():
    """Creates a SparkSession for the test suite."""
    spark_session = (
        SparkSession.builder.master("local[2]")
        .appName("pytest-pyspark-etl-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()


def test_add_age_and_drinking_status_for_adult_over_21(spark):
    """
    Tests that a person well over 21 is correctly identified with age and canDrink=True.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    # Dynamically calculate DOB for someone who is 33 years old today.
    today = date.today()
    dob_33_years_ago = today - relativedelta(years=33)
    data = [(1, dob_33_years_ago)]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_drinking_status(input_df)

    expected_data = [Row(id=1, dob=dob_33_years_ago, age=33, canDrink=True)]
    assert result_df.collect() == expected_data


def test_add_age_and_drinking_status_for_person_exactly_21(spark):
    """
    Tests the boundary case where a person is exactly 21 years old.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    # Dynamically calculate DOB for someone who turns 21 today.
    today = date.today()
    dob_exactly_21 = today - relativedelta(years=21)
    data = [(1, dob_exactly_21)]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_drinking_status(input_df)

    expected_data = [Row(id=1, dob=dob_exactly_21, age=21, canDrink=True)]
    assert result_df.collect() == expected_data


def test_add_age_and_drinking_status_for_person_just_under_21(spark):
    """
    Tests the boundary case where a person is one day shy of their 21st birthday.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    # Dynamically calculate DOB for someone who turns 21 tomorrow.
    today = date.today()
    dob_almost_21 = today - relativedelta(years=21) + relativedelta(days=1)
    data = [(1, dob_almost_21)]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_drinking_status(input_df)

    expected_data = [Row(id=1, dob=dob_almost_21, age=20, canDrink=False)]
    assert result_df.collect() == expected_data


def test_add_age_and_drinking_status_for_minor_under_21(spark):
    """
    Tests that a person well under 21 is correctly identified as a minor who cannot drink.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    # Dynamically calculate DOB for someone who is 13 years old today.
    today = date.today()
    dob_13_years_ago = today - relativedelta(years=13)
    data = [(1, dob_13_years_ago)]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_drinking_status(input_df)

    expected_data = [Row(id=1, dob=dob_13_years_ago, age=13, canDrink=False)]
    assert result_df.collect() == expected_data


def test_add_age_and_drinking_status_with_null_dob(spark):
    """
    Tests the edge case where the 'dob' is null; 'age' should be null and 'canDrink' should be False.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    data = [(1, None)]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_drinking_status(input_df)

    expected_data = [Row(id=1, dob=None, age=None, canDrink=False)]
    assert result_df.collect() == expected_data


def test_add_age_and_drinking_status_with_duplicates(spark):
    """
    Tests that duplicate input rows are processed correctly and are preserved in the output.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    today = date.today()
    dob_over_21 = today - relativedelta(years=28)
    dob_under_21 = today - relativedelta(years=18)
    data = [
        (1, "Alice", dob_over_21),
        (1, "Alice", dob_over_21),  # Duplicate
        (2, "Bob", dob_under_21),
    ]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_drinking_status(input_df)

    expected_data = {
        Row(id=1, name="Alice", dob=dob_over_21, age=28, canDrink=True),
        Row(id=2, name="Bob", dob=dob_under_21, age=18, canDrink=False),
    }
    # Check count to ensure duplicates were kept
    assert result_df.count() == 3
    # Use distinct to check the unique transformed rows
    assert set(result_df.distinct().collect()) == expected_data


def test_add_age_and_drinking_status_maintains_original_columns(spark):
    """
    Tests that the function preserves all original columns in the DataFrame.
    """
    schema = StructType([
        StructField("customer_id", StringType(), True),
        StructField("first_name", StringType(), True),
        StructField("last_name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    data = [("C1", "John", "Doe", date(2000, 1, 1))]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_drinking_status(input_df)

    expected_columns = ["customer_id", "first_name", "last_name", "dob", "age", "canDrink"]
    assert result_df.columns == expected_columns


def test_add_age_and_drinking_status_output_schema(spark):
    """
    Tests that the new columns 'age' and 'canDrink' have the correct data types.
    """
    schema = StructType([StructField("dob", DateType(), True)])
    data = [(date(2000, 1, 1),)]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_drinking_status(input_df)
    result_schema = result_df.schema

    assert result_schema["age"].dataType == IntegerType()
    assert result_schema["canDrink"].dataType == BooleanType()


def test_add_age_and_drinking_status_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the required 'dob' column is not present.
    """
    schema = StructType([StructField("id", IntegerType(), True)])
    data = [(1,)]
    input_df = spark.createDataFrame(data, schema)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_drinking_status(input_df)


@patch("main.get_spark_session")
@patch("main.read_table")
@patch("main.write_table")
def test_main_pipeline_flow(mock_write_table, mock_read_table, mock_get_spark_session, spark):
    """
    Tests the main function's orchestration logic using mocks for I/O functions.
    """
    # 1. Setup Mock Behavior
    mock_get_spark_session.return_value = spark

    today = date.today()
    dob_over_21 = today - relativedelta(years=38)
    dob_under_21 = today - relativedelta(years=9)

    input_schema = StructType([
        StructField("customer_id", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    input_data = [
        ("C1", dob_over_21),
        ("C2", dob_under_21),
    ]
    input_df = spark.createDataFrame(input_data, input_schema)
    mock_read_table.return_value = input_df
    mock_write_table.return_value = None

    # 2. Execute the main function
    main()

    # 3. Assertions
    mock_get_spark_session.assert_called_once_with("CustomerDrinkingStatus")
    mock_read_table.assert_called_once_with(spark, "gap_retail.customers")
    mock_write_table.assert_called_once()

    # Check the arguments passed to write_table
    written_df, written_table_name = mock_write_table.call_args[0]

    assert written_table_name == "gap_retail.customers"

    # Verify the content of the DataFrame passed to write_table
    expected_data = [
        Row(customer_id="C1", dob=dob_over_21, age=38, canDrink=True),
        Row(customer_id="C2", dob=dob_under_21, age=9, canDrink=False),
    ]

    # Sort by a key to ensure order for reliable comparison
    actual_data = sorted(written_df.collect(), key=lambda r: r["customer_id"])
    assert actual_data == expected_data
