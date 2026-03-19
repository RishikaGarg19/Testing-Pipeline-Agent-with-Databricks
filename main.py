import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# This script is designed to be run on a Databricks cluster.
# The environment variables are standard for connecting to Databricks APIs.
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

def create_spark_session(app_name: str) -> SparkSession:
    """
    Gets the existing SparkSession on Databricks or creates a new one.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def read_table(spark: SparkSession, table_name: str) -> DataFrame:
    """
    Reads a Delta table from the Databricks metastore.
    
    Args:
        spark: The active SparkSession.
        table_name: The fully qualified name of the table (e.g., 'schema.table').
        
    Returns:
        A DataFrame containing the data from the table.
    """
    print(f"Reading data from '{table_name}'...")
    return spark.read.table(table_name)

def add_age_and_driving_status(df: DataFrame) -> DataFrame:
    """
    Calculates age from a 'dob' column and adds an 'age' and 'canDrive' column.
    
    Args:
        df: The input DataFrame, which must contain a 'dob' column of DateType.
        
    Returns:
        A new DataFrame with the 'age' and 'canDrive' columns added.

    Raises:
        ValueError: If the 'dob' column is not present in the input DataFrame.
    """
    print("Transforming data to add 'age' and 'canDrive' columns.")

    # Check if the required 'dob' column exists.
    if "dob" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'dob' column.")

    # Calculate the age by finding the number of months between the current date and date of birth,
    # dividing by 12, and taking the floor to get the number of full years.
    df_with_age = df.withColumn(
        "age",
        F.floor(F.months_between(F.current_date(), F.col("dob")) / 12).cast(IntegerType())
    )

    # Add a 'canDrive' column based on whether the age is 18 or older.
    # This uses a conditional when/otherwise expression.
    df_with_can_drive = df_with_age.withColumn(
        "canDrive",
        F.when(F.col("age") >= 18, "Yes").otherwise("No")
    )
    
    return df_with_can_drive

def write_table(df: DataFrame, table_name: str, num_partitions: int = 4) -> None:
    """
    Writes a DataFrame to a Delta table, overwriting the existing table and schema.
    
    Args:
        df: The DataFrame to write.
        table_name: The fully qualified name of the destination table.
        num_partitions: The number of partitions to repartition the data into before writing.
    """
    print(f"Writing data to '{table_name}'...")
    # Repartitioning before a write can help control the size and number of files,
    # which is beneficial for performance of subsequent reads.
    # The 'overwriteSchema' option is set to true to allow for the new columns to be added.
    (df.repartition(num_partitions)
       .write
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(table_name))
    print(f"Successfully wrote data to '{table_name}'.")

def main():
    """
    The main function to orchestrate the ETL (Extract, Transform, Load) process.
    """
    # --- Configuration ---
    app_name = "CustomerAgeAndDrivingStatus"
    customer_table = "gap_retail.customers"

    # --- Execution ---
    # 1. Create a SparkSession
    spark = create_spark_session(app_name)

    # 2. Extract: Read the customer data
    customers_df = read_table(spark, customer_table)

    # 3. Transform: Add age and driving status columns
    transformed_customers_df = add_age_and_driving_status(customers_df)

    # 4. Load: Write the transformed data back to the table
    write_table(transformed_customers_df, customer_table)

    print("Pipeline completed successfully.")
    spark.stop()

if __name__ == "__main__":
    main()
