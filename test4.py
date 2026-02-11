import os
from pyspark.sql import SparkSession
from databricks import sql

# Create Spark session
spark = SparkSession.builder.appName("PipelineApp").getOrCreate()

# Function to read delta tables
def read_delta_table(table_name):
    return spark.read.table(table_name)

# Function to join two dataframes
def join_dataframes(df1, df2):
    return df1.join(df2, df1.id == df2.customer_id, 'inner')  # Customize the join condition

# Function to write the result to gold layer
def write_to_gold_layer(df, gold_table_name):
    df.coalesce(1).write.mode("overwrite").saveAsTable(gold_table_name)

# Main function to perform the operations
def main():
    table1 = "gap_retail.customers"  # Update with actual table name 1
    table2 = "gap_retail.orders"  # Update with actual table name 2
    gold_table = "gap_retail.gold_orders"
    df1 = read_delta_table(table1)
    df2 = read_delta_table(table2)
    joined_df = join_dataframes(df1, df2)
    write_to_gold_layer(joined_df, gold_table)

# Execute the main function
if __name__ == "__main__":
    main()