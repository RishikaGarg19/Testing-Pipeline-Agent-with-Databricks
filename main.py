import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, current_date, floor, months_between, when
from pyspark.sql.types import IntegerType

# --- Constants ---
SOURCE_SCHEMA = "gap_retail"
SOURCE_TABLE = "customers"
DESTINATION_SCHEMA = "gap_retail"
DESTINATION_TABLE = "customers_new"
MIN_DRIVING_AGE = 16
MIN_VOTING_AGE = 18


def get_spark_session() -> SparkSession:
    """Initializes and returns a SparkSession for the application."""
    # This is the standard way to create a Spark session in a Databricks environment.
    # It automatically uses the cluster's configuration.
    return SparkSession.builder.appName("CustomerEligibilityPipeline").getOrCreate()

def read_data(spark: SparkSession, schema: str, table_name: str) -> DataFrame:
    """Reads data from a specified Databricks table.

    Args:
        spark: The active SparkSession.
        schema: The schema (database) name of the source table.
        table_name: The name of the source table.

    Returns:
        A DataFrame containing the source data.
    """
    full_table_name = f"{schema}.{table_name}"
    print(f"Reading from table: {full_table_name}")
    # Reads the full table from the specified catalog and schema.
    return spark.read.table(full_table_name)

def add_age_and_eligibility_columns(df: DataFrame, driving_age: int, voting_age: int) -> DataFrame:
    """
    Adds 'age', 'canDrive', and 'canVote' columns based on the 'dob' column.

    Args:
        df: Input DataFrame, must contain a 'dob' column of DateType.
        driving_age: The minimum age to be eligible to drive.
        voting_age: The minimum age to be eligible to vote.

    Returns:
        A new DataFrame with the added 'age', 'canDrive', and 'canVote' columns.
    
    Raises:
        ValueError: If the 'dob' column is not present in the input DataFrame.
    """
    # Validate that the required 'dob' column exists.
    if "dob" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'dob' column.")
    
    # Calculate age in years from the date of birth ('dob').
    # This method is robust and correctly handles leap years and different month lengths.
    df_with_age = df.withColumn(
        "age",
        floor(months_between(current_date(), col("dob")) / 12).cast(IntegerType())
    )
    
    # Add eligibility columns based on the calculated age.
    # The when/otherwise construct correctly handles null ages by resulting in False.
    df_with_eligibility = df_with_age.withColumn(
        "canDrive",
        when(col("age") >= driving_age, True).otherwise(False)
    ).withColumn(
        "canVote",
        when(col("age") >= voting_age, True).otherwise(False)
    )
    
    return df_with_eligibility

def write_data(df: DataFrame, schema: str, table_name: str):
    """Writes the DataFrame to a specified Databricks table.

    Args:
        df: The DataFrame to be written.
        schema: The schema (database) name of the destination table.
        table_name: The name of the destination table.
    """
    # Repartitioning can help control the size and number of output files.
    # This can improve write performance and subsequent read performance on the table.
    # A value like 200 is a common starting point, but should be tuned based on data size and cluster config.
    df_repartitioned = df.repartition(200)

    full_table_name = f"{schema}.{table_name}"
    print(f"Writing to table: {full_table_name}")
    
    # Write the data, overwriting the table if it already exists.
    # Using saveAsTable registers it in the metastore, making it immediately queryable.
    df_repartitioned.write.mode("overwrite").saveAsTable(full_table_name)

def main():
    """The main function to orchestrate the ETL pipeline."""
    spark = None
    try:
        # Initialize Spark Session
        spark = get_spark_session()
        
        # Read the source customer data
        customers_df = read_data(spark, SOURCE_SCHEMA, SOURCE_TABLE)
        
        # Apply business logic to determine eligibility
        eligible_customers_df = add_age_and_eligibility_columns(
            customers_df, 
            MIN_DRIVING_AGE, 
            MIN_VOTING_AGE
        )
        
        # Write the transformed data to the new destination table
        write_data(eligible_customers_df, DESTINATION_SCHEMA, DESTINATION_TABLE)
        
        print(f"Pipeline finished successfully. Data written to {DESTINATION_SCHEMA}.{DESTINATION_TABLE}")
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()