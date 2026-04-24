import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, BooleanType
from pyspark.sql.utils import AnalysisException

# This script is designed to be run on a Databricks cluster.
# It assumes that the necessary Databricks environment variables
# (DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN)
# are configured in the execution environment, which is standard for Databricks jobs.

def get_spark_session(app_name: str) -> SparkSession:
    """
    Gets the existing SparkSession on Databricks or creates a new one.

    Args:
        app_name: The name for the Spark application.

    Returns:
        An active SparkSession instance.
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

def add_age_and_eligibility_columns(df: DataFrame) -> DataFrame:
    """
    Calculates age from a 'dob' column and adds 'age', 'canDrive', and 'canVote' columns.
    
    Args:
        df: The input DataFrame, which must contain a 'dob' column of DateType.
        
    Returns:
        A new DataFrame with the 'age', 'canDrive', and 'canVote' columns added.

    Raises:
        ValueError: If the 'dob' column is not present in the input DataFrame.
    """
    print("Transforming data to add 'age', 'canDrive', and 'canVote' columns.")

    if "dob" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'dob' column.")

    df_with_age = df.withColumn(
        "age",
        F.floor(F.months_between(F.current_date(), F.col("dob")) / 12).cast(IntegerType())
    )

    df_with_eligibility = df_with_age.withColumn(
        "canDrive",
        F.when(F.col("age") >= 16, True).otherwise(False).cast(BooleanType())
    ).withColumn(
        "canVote",
        F.when(F.col("age") >= 18, True).otherwise(False).cast(BooleanType())
    )
    
    return df_with_eligibility

def write_table(df: DataFrame, table_name: str, num_partitions: int = 4) -> None:
    """
    Writes a DataFrame to a table, overwriting any existing data.
    
    Args:
        df: The DataFrame to write.
        table_name: The fully qualified name of the destination table.
        num_partitions: The number of partitions to repartition the data into before writing.
    """
    print(f"Writing data to '{table_name}'...")
    # Repartitioning helps control the size and number of output files,
    # which can improve the performance of subsequent read operations.
    # The 'overwriteSchema' option allows for adding new columns to the table if it exists.
    (df.repartition(num_partitions)
       .write
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(table_name))
    print(f"Successfully wrote data to '{table_name}'.")

def main():
    """
    The main function to orchestrate the ETL pipeline.
    """
    app_name = "CustomerEligibilityPipeline"
    source_table = "gap_retail.customers"
    destination_table = "gap_retail.customers2"

    spark = get_spark_session(app_name)

    try:
        customers_df = read_table(spark, source_table)

        transformed_customers_df = add_age_and_eligibility_columns(customers_df)

        write_table(transformed_customers_df, destination_table)

        print("Pipeline completed successfully.")
    except (AnalysisException, ValueError) as e:
        print(f"An error occurred during the pipeline execution: {e}")
    finally:
        # It's good practice to stop the SparkSession to release resources,
        # especially in standalone script submissions.
        spark.stop()

if __name__ == "__main__":
    main()