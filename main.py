from src.etl.pipeline import run as run_etl_pipeline
from src.topic_modeling.pipeline import run_train_and_transform as run_topic_pipeline


if __name__ == "__main__":
    # Step 1: Run the ETL pipeline to process the XML and populate the database
    run_etl_pipeline()

    # Step 2: Run the topic modeling pipeline to train the model and generate topics
    run_topic_pipeline()