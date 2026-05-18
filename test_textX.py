import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DateType,
    BooleanType,
    Row
)
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

from main import add_age_and_drinking_status

@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()

def test_add_age_and_drinking_status_adult_can_drink(spark):
    """
    Tests that a person older than 21 is correctly identified as being able to drink.
    """
    today = date.today()
    dob_25_years_ago = today - relativedelta(years=25)
    
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True)
    ])
    source_data = [(1, dob_25_years_ago)]
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    result_df = add_age_and_drinking_status(source_df)

    expected_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrink", BooleanType(), False) # False nullability because otherwise() is used
    ])
    expected_data = [(1, dob_25_years_ago, 25, True)]
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_drinking_status_minor_cannot_drink(spark):
    """
    Tests that a person younger than 21 is correctly identified as not being able to drink.
    """
    today = date.today()
    dob_18_years_ago = today - relativedelta(years=18)

    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True)
    ])
    source_data = [(1, dob_18_years_ago)]
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    result_df = add_age_and_drinking_status(source_df)

    expected_data = [(1, dob_18_years_ago, 18, False)]
    
    # Using result_df.schema because it is derived from the transformation
    expected_df = spark.createDataFrame(data=expected_data, schema=result_df.schema)
    
    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_drinking_status_boundary_exactly_21(spark):
    """
    Tests the boundary case where a person is exactly 21 years old.
    """
    today = date.today()
    dob_21_years_ago = today - relativedelta(years=21)

    source_data = [(1, dob_21_years_ago)]
    source_df = spark.createDataFrame(source_data, ["id", "dob"])

    result_df = add_age_and_drinking_status(source_df)

    expected_data = [Row(id=1, dob=dob_21_years_ago, age=21, canDrink=True)]
    
    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_data)

def test_add_age_and_drinking_status_boundary_day_before_21st_birthday(spark):
    """
    Tests the boundary case where a person is one day shy of their 21st birthday.
    """
    today = date.today()
    dob_almost_21_years_ago = today - relativedelta(years=21) + timedelta(days=1)

    source_data = [(1, dob_almost_21_years_ago)]
    source_df = spark.createDataFrame(source_data, ["id", "dob"])

    result_df = add_age_and_drinking_status(source_df)

    expected_data = [Row(id=1, dob=dob_almost_21_years_ago, age=20, canDrink=False)]
    
    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_data)

def test_add_age_and_drinking_status_null_dob(spark):
    """
    Tests that a null 'dob' results in a null 'age' and 'canDrink' status of False.
    """
    source_data = [(1, None)]
    source_df = spark.createDataFrame(source_data, ["id", "dob"])

    result_df = add_age_and_drinking_status(source_df)

    expected_data = [Row(id=1, dob=None, age=None, canDrink=False)]
    
    assert result_df.count() == 1
    # Collect and compare directly as sorting can fail with mixed None/non-None types
    assert result_df.collect() == expected_data

def test_add_age_and_drinking_status_multiple_rows(spark):
    """
    Tests the transformation on a DataFrame with multiple rows of varying ages, including nulls and leap year dates.
    """
    today = date.today()
    dob_30 = today - relativedelta(years=30)
    dob_20 = today - relativedelta(years=20)
    dob_leap = date(2000, 2, 29)
    age_leap = relativedelta(today, dob_leap).years

    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True)
    ])
    source_data = [
        (1, "Alice", dob_30),    # Can drink
        (2, "Bob", dob_20),      # Cannot drink
        (3, "Charlie", None),    # Cannot drink
        (4, "Diana", dob_leap)   # Can drink
    ]
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    result_df = add_age_and_drinking_status(source_df)

    expected_data = [
        Row(id=1, name="Alice", dob=dob_30, age=30, canDrink=True),
        Row(id=2, name="Bob", dob=dob_20, age=20, canDrink=False),
        Row(id=3, name="Charlie", dob=None, age=None, canDrink=False),
        Row(id=4, name="Diana", dob=dob_leap, age=age_leap, canDrink=True)
    ]

    assert result_df.count() == 4
    assert sorted(result_df.collect(), key=lambda r: r.id) == sorted(expected_data, key=lambda r: r.id)

def test_add_age_and_drinking_status_no_dob_column_raises_error(spark):
    """
    Tests that a ValueError is raised if the 'dob' column is missing from the input DataFrame.
    """
    source_data = [(1, "some_other_data")]
    source_df = spark.createDataFrame(source_data, ["id", "not_a_dob_column"])

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_drinking_status(source_df)

def test_add_age_and_drinking_status_handles_duplicates(spark):
    """
    Tests that duplicate input rows are processed and result in duplicate output rows.
    """
    today = date.today()
    dob_40 = today - relativedelta(years=40)
    
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True)
    ])
    source_data = [
        (1, dob_40),
        (1, dob_40)  # Duplicate row
    ]
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    result_df = add_age_and_drinking_status(source_df)

    expected_data = [
        Row(id=1, dob=dob_40, age=40, canDrink=True),
        Row(id=1, dob=dob_40, age=40, canDrink=True)
    ]

    assert result_df.count() == 2
    assert sorted(result_df.collect()) == sorted(expected_data)

def test_add_age_and_drinking_status_preserves_other_columns(spark):
    """
    Tests that other columns in the source DataFrame are preserved in the output.
    """
    today = date.today()
    dob_50 = today - relativedelta(years=50)

    source_schema = StructType([
        StructField("customer_id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("city", StringType(), True)
    ])
    source_data = [(101, "John Doe", dob_50, "New York")]
    source_df = spark.createDataFrame(data=source_data, schema=source_schema)

    result_df = add_age_and_drinking_status(source_df)

    expected_data = [Row(customer_id=101, name="John Doe", dob=dob_50, city="New York", age=50, canDrink=True)]

    assert result_df.count() == 1
    assert "customer_id" in result_df.columns
    assert "name" in result_df.columns
    assert "city" in result_df.columns
    assert result_df.collect() == expected_data