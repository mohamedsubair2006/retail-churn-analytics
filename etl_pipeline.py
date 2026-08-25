import os
import duckdb
import pandas as pd

def run_sql_etl():
    print("Loading raw dataset...")
    
    # Get current folder path dynamically
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search for candidate file names automatically
    possible_files = [
        os.path.join(base_dir, 'OnlineRetail.csv'),
        os.path.join(base_dir, 'online_retail.csv'),
        os.path.join(base_dir, 'OnlineRetail.xlsx'),
        os.path.join(base_dir, 'Online Retail.xlsx')
    ]
    
    target_file = None
    for f in possible_files:
        if os.path.exists(f):
            target_file = f
            break
            
    if not target_file:
        print(f"ERROR: Could not find any retail file in: {base_dir}")
        print("Files present in directory:", os.listdir(base_dir))
        return

    print(f"Found file: {target_file}")
    
    # Read CSV or Excel depending on extension
    if target_file.endswith('.xlsx'):
        raw_data = pd.read_excel(target_file)
    else:
        raw_data = pd.read_csv(target_file, encoding='ISO-8859-1')
    
    # Standardize column names (strips accidental whitespace)
    raw_data.columns = raw_data.columns.str.strip()

    # Parse dates using pandas
    raw_data['InvoiceDate'] = pd.to_datetime(raw_data['InvoiceDate'])

    # DuckDB Execution
    con = duckdb.connect(database=':memory:')
    con.register('raw_sales', raw_data)

    print("Executing SQL Feature Engineering Pipeline...")
    sql_query = """
    WITH CleanData AS (
        SELECT 
            CAST(InvoiceNo AS STRING) AS InvoiceNo,
            CAST(CustomerID AS INT) AS CustomerID,
            InvoiceDate,
            Quantity,
            UnitPrice,
            (Quantity * UnitPrice) AS TotalLineAmount
        FROM raw_sales
        WHERE CustomerID IS NOT NULL 
          AND Quantity > 0 
          AND UnitPrice > 0
    ),
    CustomerRFM AS (
        SELECT 
            CustomerID,
            COUNT(DISTINCT InvoiceNo) AS frequency,
            SUM(TotalLineAmount) AS monetary_value,
            AVG(TotalLineAmount) AS avg_basket_size,
            MAX(InvoiceDate) AS last_purchase_date,
            MIN(InvoiceDate) AS first_purchase_date
        FROM CleanData
        GROUP BY CustomerID
    )
    SELECT 
        CustomerID,
        frequency,
        ROUND(monetary_value, 2) AS monetary_value,
        ROUND(avg_basket_size, 2) AS avg_basket_size,
        DATE_DIFF('day', last_purchase_date, TIMESTAMP '2011-12-10 00:00:00') AS recency_days,
        DATE_DIFF('day', first_purchase_date, TIMESTAMP '2011-12-10 00:00:00') AS customer_tenure_days,
        CASE 
            WHEN DATE_DIFF('day', last_purchase_date, TIMESTAMP '2011-12-10 00:00:00') > 90 THEN 1 
            ELSE 0 
        END AS is_churned
    FROM CustomerRFM;
    """

    processed_df = con.execute(sql_query).df()
    output_path = os.path.join(base_dir, 'processed_customer_features.csv')
    processed_df.to_csv(output_path, index=False)
    print(f"Success! Created '{output_path}'")

if __name__ == '__main__':
    run_sql_etl()