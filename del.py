import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, BooleanType

def get_spark_session(app_name: str) -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()

def read_table(spark: SparkSession, table_name: str) -> DataFrame:
    print(f"Reading data from '{table_name}'...")
    return spark.read.table(table_name)

def add_age_and_voting_status(df: DataFrame) -> DataFrame:
    print("Transforming data to add 'age' and 'canVote' columns.")
    if "dob" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'dob' column.")
    df_with_age = df.withColumn(
        "age",
        F.floor(F.months_between(F.current_date(), F.col("dob")) / 12).cast(IntegerType())
    )
    df_with_can_vote = df_with_age.withColumn(
        "canVote",
        F.when(F.col("age") >= 18, True).otherwise(False).cast(BooleanType())
    )
    return df_with_can_vote

def write_table(df: DataFrame, table_name: str, num_partitions: int = 4) -> None:
    print(f"Writing data to '{table_name}'...")
    (df.repartition(num_partitions)
       .write
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(table_name))
    print(f"Successfully wrote data to '{table_name}'.")

def main():
    app_name = "CustomerVotingStatusPipeline"
    source_table = "gap_retail.customers"
    destination_table = "gap_retail.customers_tr"
    spark = get_spark_session(app_name)
    try:
        customers_df = read_table(spark, source_table)
        transformed_customers_df = add_age_and_voting_status(customers_df)
        write_table(transformed_customers_df, destination_table)
        print("Pipeline completed successfully.")
    except ValueError as e:
        print(f"An error occurred during the pipeline execution: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()