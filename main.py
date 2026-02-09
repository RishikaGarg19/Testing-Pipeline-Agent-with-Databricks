import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col

def create_spark_session():
    spark = SparkSession.builder.appName("PipelineApp").getOrCreate()
    return spark

def read_source_table(spark, table_name):
    df = spark.read.table(table_name)
    return df

def add_full_name_column(df):
    df_with_full_name = df.withColumn("full_name", concat(col("first_name"), col("last_name")))
    return df_with_full_name

def write_destination_table(df, table_name):
    df.coalesce(1).write.mode("overwrite").saveAsTable(table_name)

def main():
    spark = create_spark_session()
    source_table = "gap_retail.customers"
    destination_table = "gap_retail.customers_transformed"
    df_source = read_source_table(spark, source_table)
    df_with_full_name = add_full_name_column(df_source)
    write_destination_table(df_with_full_name, destination_table)

if __name__ == "__main__":
    main()