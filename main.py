from src.etl.pipeline import run as run_etl_pipeline
from src.topics_sklearn.pipeline import run as run_topic_pipeline


if __name__ == "__main__":
    # Step 1: Run the ETL pipeline to process the XML and populate the database
    # run_etl_pipeline()

    # Step 2: Run the topic modelling pipeline to generate topics and embeddings
    # # run_topic_pipeline(db_path="data/processed/dblp.db", n_clusters=8)

    import sqlite3
    conn = sqlite3.connect("data/processed/dblp.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM topics LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    cursor.execute("SELECT * FROM papers LIMIT 50")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    conn.close()