import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col

def create_spark_session():
    return SparkSession.builder.appName('PipelineApp').getOrCreate()

def read_source_table(spark, table_name):
    return spark.read.table(table_name).coalesce(4)

def add_full_name_column(df):
    return df.withColumn('full_name', concat(col('first_name'), col('last_name'))).repartition(2)

def write_to_destination_table(df, destination_table):
    df.write.mode('overwrite').saveAsTable(destination_table)

if __name__ == '__main__':
    spark = create_spark_session()
    source_table = 'gap_retail.customers'
    destination_table = 'gap_retail.customers_transformed'
    customers_df = read_source_table(spark, source_table)
    transformed_df = add_full_name_column(customers_df)
    write_to_destination_table(transformed_df, destination_table)