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
from main import add_customer_eligibility_columns, get_spark_session


@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_get_spark_session_returns_session(spark):
    """
    Tests that get_spark_session returns a valid SparkSession instance
    with the correct application name.
    """
    app_name = "TestApp"
    test_spark = get_spark_session(app_name)
    assert isinstance(test_spark, SparkSession)
    assert test_spark.conf.get("spark.app.name") == app_name
    # Stop the session created inside the test to avoid interference
    test_spark.stop()


def test_add_customer_eligibility_columns_standard_cases(spark):
    """
    Tests that age, canDrive, and canVote are calculated correctly for
    customers over 18, under 18, and exactly 18.
    """
    today = date.today()
    schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    data = [
        (1, today - relativedelta(years=40)),  # 40 years old
        (2, today - relativedelta(years=18)),  # Exactly 18 years old today
        (3, today - relativedelta(years=17)),  # 16 years old (17th bday is today)
        (4, today - relativedelta(years=18, days=-1)), # 17 years old (18th bday is tomorrow)
    ]
    source_df = spark.createDataFrame(data, schema)

    actual_df = add_customer_eligibility_columns(source_df)

    expected_schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
        StructField("canVote", BooleanType(), True),
    ])
    expected_data = [
        (1, today - relativedelta(years=40), 40, True, True),
        (2, today - relativedelta(years=18), 18, True, True),
        (3, today - relativedelta(years=17), 17, False, False),
        (4, today - relativedelta(years=18, days=-1), 17, False, False),
    ]
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    # Sort by a unique key to ensure comparison is deterministic
    actual_rows = sorted(actual_df.collect(), key=lambda r: r['customer_id'])
    expected_rows = sorted(expected_df.collect(), key=lambda r: r['customer_id'])

    assert actual_rows == expected_rows
    assert actual_df.schema == expected_df.schema


def test_add_customer_eligibility_columns_with_null_dob(spark):
    """
    Tests that a null 'dob' results in a null 'age' and false for
    'canDrive' and 'canVote'.
    """
    schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    data = [(1, None)]
    source_df = spark.createDataFrame(data, schema)

    actual_df = add_customer_eligibility_columns(source_df)

    expected_schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
        StructField("canVote", BooleanType(), True),
    ])
    expected_data = [(1, None, None, False, False)]
    expected_df = spark.createDataFrame(expected_data, expected_schema)

    assert actual_df.collect() == expected_df.collect()
    assert actual_df.schema == expected_df.schema


def test_add_customer_eligibility_columns_empty_dataframe(spark):
    """
    Tests that the function returns an empty DataFrame with the correct schema
    when the input is empty.
    """
    schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame([], schema)

    actual_df = add_customer_eligibility_columns(source_df)

    expected_schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
        StructField("canVote", BooleanType(), True),
    ])

    assert actual_df.count() == 0
    assert actual_df.schema == expected_schema


def test_add_customer_eligibility_columns_with_duplicate_rows(spark):
    """
    Tests that duplicate input rows are preserved in the output.
    """
    today = date.today()
    schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    data = [
        (1, "Alice", today - relativedelta(years=30)),
        (1, "Alice", today - relativedelta(years=30)),  # Duplicate
        (2, "Bob", today - relativedelta(years=15)),
    ]
    source_df = spark.createDataFrame(data, schema)

    actual_df = add_customer_eligibility_columns(source_df)

    assert actual_df.count() == 3

    expected_data = [
        (1, "Alice", today - relativedelta(years=30), 30, True, True),
        (1, "Alice", today - relativedelta(years=30), 30, True, True),
        (2, "Bob", today - relativedelta(years=15), 15, False, False),
    ]

    actual_rows = sorted(actual_df.collect(), key=lambda r: (r['customer_id'], r['name']))
    expected_rows = sorted([row for row in expected_data], key=lambda r: (r[0], r[1]))

    # We need to convert tuple to Row to compare with collected data
    expected_df = spark.createDataFrame(expected_rows, actual_df.schema)
    expected_rows_as_rows = sorted(expected_df.collect(), key=lambda r: (r['customer_id'], r['name']))

    assert actual_rows == expected_rows_as_rows


def test_add_customer_eligibility_columns_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the 'dob' column is not present.
    """
    schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("date_of_birth", DateType(), True), # incorrect column name
    ])
    data = [(1, date(2000, 1, 1))]
    source_df = spark.createDataFrame(data, schema)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_customer_eligibility_columns(source_df)


def test_add_customer_eligibility_columns_edge_case_leap_year_birthday(spark):
    """
    Tests age calculation for a person born on a leap day (Feb 29).
    This ensures date logic is robust.
    """
    # Find a past leap year for the test subject
    today = date.today()
    birth_year = 2000 # a leap year
    
    # Calculate age for someone born on Feb 29, 2000
    expected_age = today.year - birth_year - ((today.month, today.day) < (2, 29))

    schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    data = [(1, date(birth_year, 2, 29))]
    source_df = spark.createDataFrame(data, schema)

    actual_df = add_customer_eligibility_columns(source_df)
    result = actual_df.first()

    assert result['age'] == expected_age
    assert result['canVote'] == (expected_age >= 18)
    assert result['canDrive'] == (expected_age >= 18)


def test_add_customer_eligibility_columns_does_not_alter_other_columns(spark):
    """
    Tests that the function adds new columns without altering existing ones.
    """
    today = date.today()
    schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    data = [
        (101, "John Doe", "john.doe@example.com", today - relativedelta(years=25)),
    ]
    source_df = spark.createDataFrame(data, schema)

    actual_df = add_customer_eligibility_columns(source_df)

    # Check that original columns are still present and unchanged
    result_row = actual_df.first()
    assert result_row['customer_id'] == 101
    assert result_row['name'] == "John Doe"
    assert result_row['email'] == "john.doe@example.com"

    # Check that new columns are added correctly
    assert "age" in actual_df.columns
    assert "canDrive" in actual_df.columns
    assert "canVote" in actual_df.columns
    assert result_row['age'] == 25
    assert result_row['canDrive'] is True
    assert result_row['canVote'] is True