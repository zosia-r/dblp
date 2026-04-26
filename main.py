from src.etl.pipeline import run as run_etl_pipeline


if __name__ == "__main__":
    # Step 1: Run the ETL pipeline to process the XML and populate the database
    run_etl_pipeline()