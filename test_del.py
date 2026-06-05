import pytest
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, DateType
from main import add_age_and_eligibility_columns

@pytest.fixture(scope='session')
def spark():
    import os
    active = SparkSession.getActiveSession()
    if active:
        return active
    if 'SPARK_REMOTE' in os.environ:
        del os.environ['SPARK_REMOTE']
    return SparkSession.builder.appName('pytest-spark-unit-tests').master('local[1]').getOrCreate()

def test_add_age_and_eligibility_columns_valid_data(spark):
    # Test with valid data
    schema = StructType([
        StructField('dob', DateType(), True)
    ])
    data = [(Row(dob='2000-01-01'),), (Row(dob='1990-06-15'),)]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_eligibility_columns(input_df)

    assert result_df.count() == 2
    assert 'age' in result_df.columns
    assert 'canDrive' in result_df.columns
    assert 'canvote' in result_df.columns
    assert result_df.filter(result_df.age == 23).count() == 1  # Adjust based on current date

def test_add_age_and_eligibility_columns_null_dob(spark):
    # Test with null dob values
    schema = StructType([
        StructField('dob', DateType(), True)
    ])
    data = [(Row(dob=None),)]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_eligibility_columns(input_df)

    assert result_df.count() == 1
    assert result_df.first().age is None
    assert result_df.first().canDrive is False
    assert result_df.first().canvote is False

def test_add_age_and_eligibility_columns_empty_dataframe(spark):
    # Test with an empty DataFrame
    schema = StructType([
        StructField('dob', DateType(), True)
    ])
    input_df = spark.createDataFrame([], schema)

    result_df = add_age_and_eligibility_columns(input_df)

    assert result_df.count() == 0

def test_add_age_and_eligibility_columns_no_dob_column(spark):
    # Test when dob column is missing
    schema = StructType([
        StructField('name', StringType(), True)
    ])
    input_df = spark.createDataFrame([], schema)

    with pytest.raises(ValueError, match='Input DataFrame must contain a dob column.'):
        add_age_and_eligibility_columns(input_df)

def test_add_age_and_eligibility_columns_boundary_age(spark):
    # Test boundary case where age is exactly 18
    schema = StructType([
        StructField('dob', DateType(), True)
    ])
    data = [(Row(dob='2005-01-01'),), (Row(dob='2004-12-31'),)]
    input_df = spark.createDataFrame(data, schema)

    result_df = add_age_and_eligibility_columns(input_df)

    assert result_df.filter(result_df.age == 18).count() == 1
    assert result_df.filter(result_df.age == 19).count() == 1
    assert result_df.filter(result_df.canDrive == True).count() == 1
    assert result_df.filter(result_df.canvote == True).count() == 1