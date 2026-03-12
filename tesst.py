import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# This script is designed to be run on a Databricks cluster.
# The SparkSession is automatically configured on Databricks.
# The environment variables are standard for connecting to Databricks APIs
# and are included here as per the requirements.
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

def create_spark_session(app_name: str) -> SparkSession:
    """Creates and returns a SparkSession."""
    # When running on Databricks, this gets the existing session.
    return SparkSession.builder.appName(app_name).getOrCreate()

def read_table(spark: SparkSession, table_name: str) -> DataFrame:
    """Reads data from a specified Databricks table."""
    print(f"Reading from table: {table_name}")
    return spark.read.table(table_name)

def transform_customers_data(df: DataFrame, dob_col: str, driving_age: int) -> DataFrame:
    """
    Adds 'age' and 'canDrive' columns to the customer DataFrame.
    
    Args:
        df: The input DataFrame containing customer data.
        dob_col: The name of the column with the date of birth.
        driving_age: The minimum legal age for driving.

    Returns:
        A new DataFrame with 'age' and 'canDrive' columns added.
    """
    # Calculate age based on the date of birth column.
    # The age is the number of full years that have passed.
    df_with_age = df.withColumn(
        "age",
        F.floor(F.months_between(F.current_date(), F.col(dob_col)) / 12).cast(IntegerType())
    )

    # Add the 'canDrive' column based on the calculated age.
    # If 'dob' is NULL, 'age' will be NULL, and 'canDrive' will be 'No'.
    df_with_drive_status = df_with_age.withColumn(
        "canDrive",
        F.when(F.col("age") >= driving_age, "Yes").otherwise("No")
    )
    
    return df_with_drive_status

def write_table(df: DataFrame, table_name: str, partitions: int = 4) -> None:
    """Writes a DataFrame to a Databricks table after repartitioning."""
    print(f"Writing data to table: {table_name}")
    # Repartition to control the number of output files for better performance.
    df.repartition(partitions).write.mode("overwrite").saveAsTable(table_name)

def main():
    """Main function to orchestrate the ETL pipeline."""
    # Configuration parameters
    input_table = "gap_retail.customers"
    output_table = "gap_retail.customers_with_drive_status"
    app_name = "DrivingStatusPipeline"
    driving_age_threshold = 18

    # 1. Initialize Spark Session
    spark = create_spark_session(app_name)

    # 2. Read source data
    customers_df = read_table(spark, input_table)

    # 3. Apply transformations
    # The original 'dob' column is retained as requested.
    final_df = transform_customers_data(customers_df, "dob", driving_age_threshold)

    # 4. Write the transformed data to a new table
    write_table(final_df, output_table)

    print(f"Pipeline finished successfully. Output written to {output_table}.")
    spark.stop()

if __name__ == "__main__":
    main()