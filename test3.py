from pyspark.sql import SparkSession
import os

# Initialize Spark session
spark = SparkSession.builder.appName("PipelineApp").getOrCreate()

# Read data from source table
source_table = "gap_retail.customers"

def read_table(table_name):
    return spark.read.table(table_name)

# Add full_name column

def add_full_name_column(df):
    return df.withColumn("full_name", concat(coalesce(df.first_name, lit('')), lit(' '), coalesce(df.last_name, lit(''))))

# Write data to destination table

def write_table(df, table_name):
    df.coalesce(1).write.mode("overwrite").saveAsTable(table_name)

# Process
customers_df = read_table(source_table)
customers_with_full_name = add_full_name_column(customers_df)
write_table(customers_with_full_name, "gap_retail.customers")