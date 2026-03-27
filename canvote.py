import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, BooleanType

# This script is designed to be run on a Databricks cluster.
# The environment variables DATABRICKS_HOST, DATABRICKS_HTTP_PATH, and DATABRICKS_TOKEN
# are typically pre-configured in Databricks environments.

def get_spark_session(app_name: str) -> SparkSession:
    """
    Gets the existing SparkSession on Databricks or creates a new one.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def read_table(spark: SparkSession, table_name: str) -> DataFrame:
    """
    Reads a table from the Databricks metastore.
    
    Args:
        spark: The active SparkSession.
        table_name: The fully qualified name of the table (e.g., 'schema.table_name').
        
    Returns:
        A DataFrame containing the data from the table.
    """
    print(f"Reading data from '{table_name}'...")
    return spark.read.table(table_name)

def add_age_and_voting_status(df: DataFrame) -> DataFrame:
    """
    Calculates age from a 'dob' column and adds 'age' and 'canVote' columns.
    
    Args:
        df: The input DataFrame, which must contain a 'dob' column of DateType.
        
    Returns:
        A new DataFrame with the 'age' and 'canVote' columns added.

    Raises:
        ValueError: If the 'dob' column is not present in the input DataFrame.
    """
    print("Transforming data to add 'age' and 'canVote' columns.")

    # Validate that the required 'dob' column exists before proceeding.
    if "dob" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'dob' column.")

    # Calculate age based on the 'dob' column.
    # The age is the number of full years passed since the date of birth.
    df_with_age = df.withColumn(
        "age",
        F.floor(F.months_between(F.current_date(), F.col("dob")) / 12).cast(IntegerType())
    )

    # Add a 'canVote' boolean column based on whether the age is 18 or greater.
    df_with_can_vote = df_with_age.withColumn(
        "canVote",
        F.when(F.col("age") >= 18, True).otherwise(False).cast(BooleanType()) # Fixed: Logic error, should be >= 18.
    )
    
    return df_with_can_vote

def write_table(df: DataFrame, table_name: str, num_partitions: int = 4) -> None:
    """
    Writes a DataFrame to a table, overwriting the existing table and schema.
    
    Args:
        df: The DataFrame to write.
        table_name: The fully qualified name of the destination table.
        num_partitions: The number of partitions to repartition the data into before writing.
    """
    print(f"Writing data to '{table_name}'...")
    # Repartitioning before writing helps control the size and number of output files,
    # which can improve the performance of subsequent read operations.
    # The 'overwriteSchema' option allows for adding the new columns to the table.
    (df.repartition(num_partitions)
       .write
       .mode("overwrite") # Fixed: This will now correctly overwrite the table on each run.
       .option("overwriteSchema", "true")
       .saveAsTable(table_name))
    print(f"Successfully wrote data to '{table_name}'.")

def main():
    """
    The main function to orchestrate the ETL pipeline.
    """
    # --- Configuration ---
    app_name = "CustomerVotingStatus"
    customer_table = "gap_retail.customers"

    # --- Execution ---
    # 1. Create a SparkSession
    spark = get_spark_session(app_name)

    # 2. Extract: Read the customer data
    customers_df = read_table(spark, customer_table)

    # 3. Transform: Add age and voting status columns
    transformed_customers_df = add_age_and_voting_status(customers_df)

    # 4. Load: Write the transformed data back to the same table
    write_table(transformed_customers_df, customer_table)

    print("Pipeline completed successfully.")
    spark.stop() # Fixed: SparkSession is now stopped to release resources.

if __name__ == "__main__":
    main()
