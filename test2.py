import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col

def create_spark_session():
    spark = SparkSession.builder.appName("PipelineApp").getOrCreate()
    return spark

def read_source_table(spark):
    return spark.read.table("gap_retail.customers")

def transform_data(df):
    return df.withColumn("full_name", concat(col("first_name"), col("last_name")))

def write_to_destination(df):
    df.coalesce(1).write.mode("overwrite").saveAsTable("gap_retail.customers_transformed")

def main():
    spark = create_spark_session()
    source_df = read_source_table(spark)
    transformed_df = transform_data(source_df)
    write_to_destination(transformed_df)

if __name__ == "__main__":
    main()