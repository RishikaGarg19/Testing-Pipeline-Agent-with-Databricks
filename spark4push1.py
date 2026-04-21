import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, current_date, floor, lit, months_between, when
from pyspark.sql.types import IntegerType

def create_spark_session() -> SparkSession:
    """Creates and returns a Spark session for the Databricks environment."""
    return SparkSession.builder.appName("CustomerAgePipeline").getOrCreate()

def read_customer_data(spark: SparkSession, table_name: str) -> DataFrame:
    """Reads customer data from a specified Databricks table."""
    print(f"Reading data from {table_name}...")
    return spark.read.table(table_name)

def add_age_and_driving_status(df: DataFrame) -> DataFrame:
    """
    Adds 'age' and 'canDrive' columns to a DataFrame based on a 'dob' column.

    The function calculates the age based on the date of birth and determines
    if the person is of legal driving age (16 or older). It is designed to pass
    the unit tests provided in the reference code.

    :param df: Input DataFrame, must contain a 'dob' column of DateType.
    :return: A new DataFrame with 'age' (IntegerType) and 'canDrive' (BooleanType) columns.
    :raises ValueError: If the input DataFrame does not contain the 'dob' column.
    """
    if 'dob' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'dob' column.")

    # Calculate age in full years using months_between for accuracy.
    df_with_age = df.withColumn(
        'age',
        floor(months_between(current_date(), col('dob')) / 12).cast(IntegerType())
    )

    # Determine driving status. Ages >= 16 can drive.
    # The otherwise(lit(False)) handles null ages, setting 'canDrive' to False
    # and ensures the column is non-nullable, matching the unit test requirements.
    df_with_driving_status = df_with_age.withColumn(
        'canDrive',
        when(col('age') >= 16, lit(True)).otherwise(lit(False))
    )
    
    return df_with_driving_status

def write_data(df: DataFrame, table_name: str) -> None:
    """
    Writes a DataFrame to a specified Databricks table.

    The data is repartitioned before writing to optimize performance by controlling
    the number of output files. The write mode is set to 'overwrite'.

    :param df: The DataFrame to be written.
    :param table_name: The full name of the destination table (e.g., 'schema.table_name').
    """
    print(f"Writing data to {table_name}...")
    # Repartition to control the number of output files and improve write performance.
    num_partitions = 8
    df.repartition(num_partitions).write.mode("overwrite").saveAsTable(table_name)
    print("Write operation completed.")

def main():
    """Main function to orchestrate the ETL pipeline."""
    spark = create_spark_session()
    
    source_table = "gap_retail.customers"
    # A new table is used for the destination to avoid overwriting the source data.
    destination_table = "gap_retail.customers_with_age"

    try:
        # Step 1: Read data from the source table.
        customers_df = read_customer_data(spark, source_table)
        
        # Step 2: Apply the core transformation to add age and driving status.
        transformed_df = add_age_and_driving_status(customers_df)
        
        # Step 3: Write the transformed data to the destination table.
        write_data(transformed_df, destination_table)
        
    finally:
        # Ensure the Spark session is stopped even if an error occurs.
        spark.stop()

if __name__ == "__main__":
    main()
