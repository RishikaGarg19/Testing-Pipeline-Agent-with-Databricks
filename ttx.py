from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col
import os

spark = SparkSession.builder.appName("PipelineApp").getOrCreate()

def connect_databricks_sql():
    try:
        from databricks import sql
    except ImportError:
        raise ImportError("databricks.sql is not available in this environment")
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )
    return connection

def read_bronze_orders():
    return spark.read.table("bronze_orders")

def deduplicate_orders(df):
    window_spec = Window.partitionBy("order_id").orderBy(col("ingestion_ts").desc())
    deduped_df = df.withColumn("rn", row_number().over(window_spec)).filter(col("rn") == 1).drop("rn")
    return deduped_df

def write_idempotent_merge(df):
    df_coalesced = df.coalesce(1)
    target_table = "silver_orders"
    spark.sql(f"CREATE TABLE IF NOT EXISTS {target_table} USING DELTA AS SELECT * FROM {target_table} WHERE 1=0")
    df_coalesced.createOrReplaceTempView("updates")
    merge_sql = f"""
        MERGE INTO {target_table} AS t
        USING updates AS s
        ON t.order_id = s.order_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """
    spark.sql(merge_sql)

def main():
    bronze_df = read_bronze_orders()
    dedup_df = deduplicate_orders(bronze_df)
    write_idempotent_merge(dedup_df)
    spark.stop()

if __name__ == "__main__":
    main()