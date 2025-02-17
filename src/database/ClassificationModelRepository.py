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
        
    def saving_classification_model(self, model_name,model_pickle,website):
        classification_model = Table('classification_model', self.metadata, autoload_with=self.engine)

        with open(model_pickle, "rb") as f:
            model_data = f.read()
        with self.engine.connect() as conn:
            with conn.begin():
                stmt = sa.insert(classification_model).values(model_name = model_name,model_data=model_data, website = website)
                conn.execute(stmt)
        logging.debug(f"model is saved")

    def get_classification_model(self,model_name):
        classification_model = Table('classification_model', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(classification_model.c.model_data).where(classification_model.c.model_name == model_name)
            result = conn.execute(stmt)

            model_path = "models/loaded_model.pth"
            with open(model_path, "wb") as f:
                f.write(result)
            