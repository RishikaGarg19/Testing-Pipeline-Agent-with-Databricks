import pytest
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from main import deduplicate_orders

@pytest.fixture(scope='session')
def spark():
    import os
    builder = SparkSession.builder.appName('pytest-spark-unit-tests')
    if 'SPARK_REMOTE' not in os.environ:
        builder = builder.master('local[1]')
    return builder.getOrCreate()

def test_deduplicate_orders_keeps_latest_record_per_order_id(spark):
    now = datetime.now()
    data = [
        ("order1", now - timedelta(days=1)),
        ("order1", now),
        ("order2", now - timedelta(days=2)),
        ("order2", now),
    ]
    df = spark.createDataFrame(data, ["order_id", "ingestion_ts"])
    result_df = deduplicate_orders(df)
    result = result_df.collect()
    assert len(result) == 2
    assert set(r.order_id for r in result) == {"order1", "order2"}
    for row in result:
        if row.order_id == "order1":
            assert row.ingestion_ts == now
        if row.order_id == "order2":
            assert row.ingestion_ts == now

def test_deduplicate_orders_handles_null_timestamps(spark):
    now = datetime.now()
    data = [
        ("order1", None),
        ("order1", now),
    ]
    df = spark.createDataFrame(data, ["order_id", "ingestion_ts"])
    result_df = deduplicate_orders(df)
    result = result_df.collect()
    assert len(result) == 1
    assert result[0].order_id == "order1"
    assert result[0].ingestion_ts == now

def test_deduplicate_orders_with_duplicate_rows_same_timestamp(spark):
    now = datetime.now()
    data = [
        ("order1", now),
        ("order1", now),
    ]
    df = spark.createDataFrame(data, ["order_id", "ingestion_ts"])
    result_df = deduplicate_orders(df)
    result = result_df.collect()
    # Only one record should remain even if timestamps are identical
    assert len(result) == 1
    assert result[0].order_id == "order1"
    assert result[0].ingestion_ts == now

def test_deduplicate_orders_boundary_values(spark):
    min_ts = datetime(1970, 1, 1)
    max_ts = datetime(3000, 1, 1)
    data = [
        ("order1", min_ts),
        ("order1", max_ts),
    ]
    df = spark.createDataFrame(data, ["order_id", "ingestion_ts"])
    result_df = deduplicate_orders(df)
    result = result_df.collect()
    assert len(result) == 1
    assert result[0].ingestion_ts == max_ts