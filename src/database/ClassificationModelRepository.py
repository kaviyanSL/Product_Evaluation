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
        
    def saving_classification_model(self,model_pickle):
        classification_model = Table('classification_model', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            with conn.begin():
                stmt = sa.insert(classification_model).values(model=model_pickle)
                conn.execute(stmt)
        logging.debug(f"model is saved")

    def get_classification_model(self):
        classification_model = Table('classification_model', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(classification_model.c.model)
            result = conn.execute(stmt)
            return result.fetchone()