import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData
from src.database.db_connection import DBConnection
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



class UserPromptrepository:
    def __init__(self):
        self.db_connection = DBConnection()
        self.engine = self.db_connection.get_engine()
        self.metadata = MetaData()
    
    def saving_keywords(self,comment,user,keywords,generated_prompt,ip):
        usaer_prompt = Table('usaer_prompt', self.metadata, autoload_with=self.engine)

        with self.engine.connect() as conn:
            with conn.begin():
                stmt = sa.insert(usaer_prompt).values(
                    prompt_text=comment,
                    prompt_keyword=keywords,
                    user_name=user,
                    optimized_generated_prompt = generated_prompt,
                    ip_address = ip
                    )
                conn.execute(stmt)
                logging.debug("prompt are saved successfully!")

    