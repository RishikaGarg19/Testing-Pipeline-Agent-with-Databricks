import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, BooleanType
from datetime import date
from dateutil.relativedelta import relativedelta

from main import add_age_and_driving_status


@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()


def test_add_age_and_driving_status_calculates_age_and_can_drive_correctly(spark):
    """
    Tests that age and canDrive are calculated correctly for various birth dates.
    """
    today = date.today()
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    input_data = [
        Row(id=1, dob=today - relativedelta(years=25, months=6)),  # 25 years old
        Row(id=2, dob=today - relativedelta(years=15, months=11)), # 15 years old
        Row(id=3, dob=today - relativedelta(years=16)),           # Exactly 16 years old
        Row(id=4, dob=today - relativedelta(years=50)),           # 50 years old
    ]
    input_df = spark.createDataFrame(input_data, schema)

    expected_data = [
        Row(id=1, dob=today - relativedelta(years=25, months=6), age=25, canDrive=True),
        Row(id=2, dob=today - relativedelta(years=15, months=11), age=15, canDrive=False),
        Row(id=3, dob=today - relativedelta(years=16), age=16, canDrive=True),
        Row(id=4, dob=today - relativedelta(years=50), age=50, canDrive=True),
    ]
    
    result_df = add_age_and_driving_status(input_df)
    
    # Sort by id to ensure order doesn't affect the comparison
    result_rows = sorted(result_df.collect(), key=lambda r: r.id)
    expected_rows = sorted(expected_data, key=lambda r: r.id)

    assert result_rows == expected_rows


def test_add_age_and_driving_status_raises_error_if_dob_missing(spark):
    """
    Tests that a ValueError is raised if the 'dob' column is not in the input DataFrame.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("birth_date", DateType(), True),
    ])
    input_data = [
        Row(id=1, birth_date=date(2000, 1, 1))
    ]
    input_df = spark.createDataFrame(input_data, schema)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_driving_status(input_df)


def test_add_age_and_driving_status_handles_null_dob(spark):
    """
    Tests that a null 'dob' results in a null 'age' and 'canDrive' as False.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    input_data = [
        Row(id=1, dob=date(1990, 5, 15)),
        Row(id=2, dob=None),
        Row(id=3, dob=date(2010, 1, 1)),
    ]
    input_df = spark.createDataFrame(input_data, schema)

    expected_data = [
        Row(id=1, dob=date(1990, 5, 15), age=date.today().year - 1990, canDrive=True),
        Row(id=2, dob=None, age=None, canDrive=False),
        Row(id=3, dob=date(2010, 1, 1), age=date.today().year - 2010, canDrive=False),
    ]
    
    result_df = add_age_and_driving_status(input_df)
    
    # The exact age for id=1 and id=3 depends on the run date, so we check the canDrive status and nulls
    result_rows = sorted(result_df.collect(), key=lambda r: r.id)
    
    assert result_rows[0].canDrive is True
    assert result_rows[0].age is not None

    assert result_rows[1].age is None
    assert result_rows[1].canDrive is False

    assert result_rows[2].canDrive is False
    assert result_rows[2].age is not None
    

def test_add_age_and_driving_status_preserves_duplicates(spark):
    """
    Tests that duplicate input rows are preserved in the output.
    """
    today = date.today()
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    input_data = [
        Row(id=1, name="Alice", dob=today - relativedelta(years=30)),
        Row(id=2, name="Bob", dob=today - relativedelta(years=14)),
        Row(id=2, name="Bob", dob=today - relativedelta(years=14)), # Duplicate
    ]
    input_df = spark.createDataFrame(input_data, schema)

    result_df = add_age_and_driving_status(input_df)

    assert result_df.count() == 3
    
    # Check that the duplicate row has been transformed correctly
    bob_rows = result_df.filter("name = 'Bob'").collect()
    assert len(bob_rows) == 2
    assert bob_rows[0].age == 14
    assert bob_rows[0].canDrive is False
    assert bob_rows[1].age == 14
    assert bob_rows[1].canDrive is False


def test_add_age_and_driving_status_boundary_turns_16_tomorrow(spark):
    """
    Tests the boundary case for a person who will turn 16 tomorrow.
    """
    today = date.today()
    dob_turns_16_tomorrow = today - relativedelta(years=16) + relativedelta(days=1)
    
    schema = StructType([StructField("dob", DateType(), True)])
    input_df = spark.createDataFrame([Row(dob=dob_turns_16_tomorrow)], schema)
    
    result_df = add_age_and_driving_status(input_df)
    result = result_df.first()

    assert result.age == 15
    assert result.canDrive is False


def test_add_age_and_driving_status_boundary_just_turned_16_today(spark):
    """
    Tests the boundary case for a person whose 16th birthday is today.
    """
    today = date.today()
    dob_turned_16_today = today - relativedelta(years=16)
    
    schema = StructType([StructField("dob", DateType(), True)])
    input_df = spark.createDataFrame([Row(dob=dob_turned_16_today)], schema)

    result_df = add_age_and_driving_status(input_df)
    result = result_df.first()

    assert result.age == 16
    assert result.canDrive is True


def test_add_age_and_driving_status_boundary_age_zero(spark):
    """
    Tests the case for a newborn (age 0).
    """
    today = date.today()
    dob_newborn = today - relativedelta(months=3)
    
    schema = StructType([StructField("dob", DateType(), True)])
    input_df = spark.createDataFrame([Row(dob=dob_newborn)], schema)
    
    result_df = add_age_and_driving_status(input_df)
    result = result_df.first()

    assert result.age == 0
    assert result.canDrive is False
    
    
def test_add_age_and_driving_status_handles_empty_dataframe(spark):
    """
    Tests that an empty DataFrame is handled gracefully.
    """
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("dob", DateType(), True),
    ])
    input_df = spark.createDataFrame([], schema)

    result_df = add_age_and_driving_status(input_df)

    assert result_df.count() == 0
    assert "age" in result_df.columns
    assert "canDrive" in result_df.columns


def test_add_age_and_driving_status_output_schema_is_correct(spark):
    """
    Tests that the output DataFrame has the correct schema and data types.
    """
    today = date.today()
    schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
    ])
    input_data = [Row(name="Test", dob=today - relativedelta(years=20))]
    input_df = spark.createDataFrame(input_data, schema)
    
    result_df = add_age_and_driving_status(input_df)
    
    expected_schema = StructType([
        StructField("name", StringType(), True),
        StructField("dob", DateType(), True),
        StructField("age", IntegerType(), True),
        StructField("canDrive", BooleanType(), False), # Note: 'when' makes it non-nullable
    ])

    assert result_df.schema == expected_schema