import pytest
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, DateType
from main import add_age_and_voting_status

@pytest.fixture(scope='session')
def spark():
    import os
    active = SparkSession.getActiveSession()
    if active:
        return active
    if 'SPARK_REMOTE' in os.environ:
        del os.environ['SPARK_REMOTE']
    return SparkSession.builder.appName('pytest-spark-unit-tests').master('local[1]').getOrCreate()

def test_add_age_and_voting_status_valid_data(spark):
    data = [
        Row(dob="2000-01-01"),
        Row(dob="2010-12-31"),
        Row(dob="1995-06-15")
    ]
    schema = StructType([StructField("dob", DateType(), True)])
    df = spark.createDataFrame(data, schema)
    
    result_df = add_age_and_voting_status(df).collect()
    assert result_df[0]['age'] == 23  # assuming current date is 2023-01-01
    assert result_df[0]['canVote'] is True
    assert result_df[1]['age'] == 13
    assert result_df[1]['canVote'] is False
    assert result_df[2]['age'] == 27
    assert result_df[2]['canVote'] is True

def test_add_age_and_voting_status_null_dob(spark):
    data = [
        Row(dob=None)
    ]
    schema = StructType([StructField("dob", DateType(), True)])
    df = spark.createDataFrame(data, schema)

    result_df = add_age_and_voting_status(df).collect()
    assert result_df[0]['age'] is None
    assert result_df[0]['canVote'] is False

def test_add_age_and_voting_status_empty_df(spark):
    data = []
    schema = StructType([StructField("dob", DateType(), True)])
    df = spark.createDataFrame(data, schema)
    
    result_df = add_age_and_voting_status(df)
    assert result_df.count() == 0

def test_add_age_and_voting_status_invalid_column(spark):
    data = [
        Row(name="John Doe")  # no 'dob' column
    ]
    schema = StructType([StructField("name", StringType(), True)])
    df = spark.createDataFrame(data, schema)

    with pytest.raises(ValueError, match="Input DataFrame must contain a 'dob' column."):
        add_age_and_voting_status(df)

def test_add_age_and_voting_status_duplicates(spark):
    data = [
        Row(dob="2000-01-01"),
        Row(dob="2000-01-01")
    ]
    schema = StructType([StructField("dob", DateType(), True)])
    df = spark.createDataFrame(data, schema)

    result_df = add_age_and_voting_status(df).collect()
    assert result_df[0]['age'] == 23  # assuming the current date is 2023-01-01
    assert result_df[0]['canVote'] is True
    assert result_df[1]['age'] == 23
    assert result_df[1]['canVote'] is True

def test_add_age_and_voting_status_boundary_values(spark):
    data = [
        Row(dob="2005-01-01"),  # exactly 18 years old
        Row(dob="2004-12-31"),  # just under 18
    ]
    schema = StructType([StructField("dob", DateType(), True)])
    df = spark.createDataFrame(data, schema)

    result_df = add_age_and_voting_status(df).collect()
    assert result_df[0]['age'] == 18
    assert result_df[0]['canVote'] is True
    assert result_df[1]['age'] == 18
    assert result_df[1]['canVote'] is True