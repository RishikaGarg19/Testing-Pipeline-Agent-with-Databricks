import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import round

def create_spark_session():
    spark = SparkSession.builder.appName("PipelineApp").getOrCreate()
    return spark

def read_fact_sps_pump(spark):
    df = spark.read.table("gold.fact_sps_pump")
    return df

def calculate_avg_pump_run_duration(df):
    updated_df = df.withColumn("avg_pump_run_duration", round(df.run_hrs_yd / df.start_yd, 2))
    return updated_df

def write_to_staging_table(df):
    df.write.mode("overwrite").saveAsTable("gold.stg_fact_sps_pump")

def read_staging_table(spark):
    df_staging = spark.read.table("gold.stg_fact_sps_pump")
    return df_staging

def write_back_to_original_table(df):
    df.write.mode("overwrite").saveAsTable("gold.fact_sps_pump")

def main():
    spark = create_spark_session()
    df_fact = read_fact_sps_pump(spark)
    df_calculated = calculate_avg_pump_run_duration(df_fact)
    df_calculated.show()
    write_to_staging_table(df_calculated)
    df_staging = read_staging_table(spark)
    write_back_to_original_table(df_staging)
    print("Calculation and write operations completed successfully.")

if __name__ == "__main__":
    main()