import os
import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData
from src.database.db_connection import DBConnection
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ClassificationModelRepository:
    def __init__(self):
        self.db_connection = DBConnection()
        self.engine = self.db_connection.get_engine()
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)  # Ensure metadata is loaded correctly

    def saving_classification_model(self, model_name, model_pickle, website):
        classification_model = Table('classification_model', self.metadata, autoload_with=self.engine)

        # Check if the model file exists
        if not os.path.exists(model_pickle):
            logging.error(f"Model file not found: {model_pickle}")
            return
        
        # Read the model file as binary data
        with open(model_pickle, "rb") as f:
            model_data = f.read()

        # Insert into database
        stmt = sa.insert(classification_model).values(
            model_name=model_name,
            model_data=model_data,
            website=website
        )

        # Ensure commit happens
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                conn.execute(stmt)
                trans.commit()
                logging.debug("Model saved successfully!")
            except Exception as e:
                trans.rollback()
                logging.error(f"Database insert failed: {e}")


    def get_classification_model(self,model_name):
        classification_model = Table('classification_model', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(classification_model.c.model_data).where(classification_model.c.model_name == model_name)
            result = conn.execute(stmt)

            model_path = "models/loaded_model.pth"
            with open(model_path, "wb") as f:
                f.write(result)
    
    def get_specific_comment_data(self, row_id):
        clustered_comment_table = Table("clustered_comment", self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(clustered_comment_table).where(clustered_comment_table.c.id == row_id)
            result = conn.execute(stmt).fetchone()
            return result if result else None