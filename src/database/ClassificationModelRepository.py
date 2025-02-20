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

    def saving_classification_model(self, model_name, model_data, website):
        classification_model = Table('classification_model', self.metadata, autoload_with=self.engine)

        with self.engine.connect() as conn:
            with conn.begin():
                stmt = sa.insert(classification_model).values(
                    model_name=model_name,
                    model_data=model_data,
                    website=website
                    )
                conn.execute(stmt)
                logging.debug("Model saved successfully!")



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