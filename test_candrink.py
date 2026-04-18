import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, BooleanType, LongType
from datetime import date
from dateutil.relativedelta import relativedelta

from main import add_age_and_eligibility_columns

# Per CRITICAL RULES, this fixture MUST be copied verbatim
@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()

def test_add_age_and_eligibility_columns_under_18(spark):
    """
    Tests that a person under 18 is correctly identified as unable to drive or drink.
    """
    today = date.today()
    dob_17_years_ago = today - relativedelta(years=17)

    source_data = [Row(id=1, dob=dob_17_years_ago)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [Row(id=1, dob=dob_17_years_ago, age=17, canDrive=False, canDrink=False)]
    expected_schema = StructType([
        StructField("id", LongType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), True),
        StructField("canDrink", BooleanType(), True),
    ])
    expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())
    assert result_df.schema == expected_df.schema

def test_add_age_and_eligibility_columns_is_18(spark):
    """
    Tests that a person who is exactly 18 can drive but not drink.
    """
    today = date.today()
    dob_18_years_ago = today - relativedelta(years=18)

    source_data = [Row(id=1, dob=dob_18_years_ago)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [Row(id=1, dob=dob_18_years_ago, age=18, canDrive=True, canDrink=False)]
    expected_df = spark.createDataFrame(expected_data)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_between_18_and_21(spark):
    """
    Tests that a person between 18 and 21 (e.g., 20) can drive but not drink.
    """
    today = date.today()
    dob_20_years_ago = today - relativedelta(years=20)

    source_data = [Row(id=1, dob=dob_20_years_ago)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [Row(id=1, dob=dob_20_years_ago, age=20, canDrive=True, canDrink=False)]
    expected_df = spark.createDataFrame(expected_data)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_is_21(spark):
    """
    Tests that a person who is exactly 21 can both drive and drink.
    """
    today = date.today()
    dob_21_years_ago = today - relativedelta(years=21)

    source_data = [Row(id=1, dob=dob_21_years_ago)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [Row(id=1, dob=dob_21_years_ago, age=21, canDrive=True, canDrink=True)]
    expected_df = spark.createDataFrame(expected_data)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_over_21(spark):
    """
    Tests that a person over 21 (e.g., 40) can both drive and drink.
    """
    today = date.today()
    dob_40_years_ago = today - relativedelta(years=40)

    source_data = [Row(id=1, dob=dob_40_years_ago)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [Row(id=1, dob=dob_40_years_ago, age=40, canDrive=True, canDrink=True)]
    expected_df = spark.createDataFrame(expected_data)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_null_dob(spark):
    """
    Tests that a row with a null 'dob' results in a null 'age' and false for eligibility flags.
    """
    source_data = [Row(id=1, dob=None)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [Row(id=1, dob=None, age=None, canDrive=False, canDrink=False)]
    expected_df = spark.createDataFrame(expected_data)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_birthday_is_today(spark):
    """
    Tests the boundary case where the date of birth is the current date, resulting in age 0.
    """
    today = date.today()

    source_data = [Row(id=1, dob=today)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [Row(id=1, dob=today, age=0, canDrive=False, canDrink=False)]
    expected_df = spark.createDataFrame(expected_data)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_birthday_is_future(spark):
    """
    Tests the edge case where the date of birth is in the future, resulting in a negative age.
    """
    today = date.today()
    future_dob = today + relativedelta(years=1)

    source_data = [Row(id=1, dob=future_dob)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [Row(id=1, dob=future_dob, age=-1, canDrive=False, canDrink=False)]
    expected_df = spark.createDataFrame(expected_data)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 1
    assert sorted(result_df.collect()) == sorted(expected_df.collect())

def test_add_age_and_eligibility_columns_preserves_other_columns(spark):
    """
    Tests that existing columns in the input DataFrame are preserved in the output.
    """
    today = date.today()
    dob = today - relativedelta(years=25)

    source_data = [Row(id=100, name='John Doe', city='New York', dob=dob)]
    source_df = spark.createDataFrame(source_data)

    expected_data = [Row(id=100, name='John Doe', city='New York', dob=dob, age=25, canDrive=True, canDrink=True)]
    expected_df = spark.createDataFrame(expected_data)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 1
    assert sorted(result_df.columns) == sorted(expected_df.columns)
    assert result_df.select('id', 'name', 'city').collect()[0] == Row(id=100, name='John Doe', city='New York')

def test_add_age_and_eligibility_columns_handles_duplicates(spark):
    """
    Tests that duplicate input rows are processed correctly.
    """
    today = date.today()
    dob_30 = today - relativedelta(years=30)
    dob_16 = today - relativedelta(years=16)

    source_data = [
        Row(id=1, dob=dob_30),
        Row(id=1, dob=dob_30), # Duplicate
        Row(id=2, dob=dob_16)
    ]
    source_df = spark.createDataFrame(source_data)

    result_df = add_age_and_eligibility_columns(source_df)

    assert result_df.count() == 3
    # Check count for a specific case
    assert result_df.filter("age = 30 AND canDrive = true AND canDrink = true").count() == 2
    assert result_df.filter("age = 16 AND canDrive = false AND canDrink = false").count() == 1

def test_add_age_and_eligibility_columns_empty_dataframe(spark):
    """
    Tests that the function handles an empty input DataFrame gracefully.
    """
    source_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    source_df = spark.createDataFrame([], schema=source_schema)

    result_df = add_age_and_eligibility_columns(source_df)
    
    expected_columns = ["id", "dob", "age", "canDrive", "canDrink"]
    assert result_df.count() == 0
    assert sorted(result_df.columns) == sorted(expected_columns)

def test_add_age_and_eligibility_columns_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the 'dob' column is not present.
    """
    source_data = [Row(id=1, date_of_birth=date(2000, 1, 1))]
    source_df = spark.createDataFrame(source_data)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_eligibility_columns(source_df)