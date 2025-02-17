import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData
from src.database.db_connection import DBConnection
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PreProcessCommentsrepository:
    def __init__(self):
        self.db_connection = DBConnection()
        self.engine = self.db_connection.get_engine()
        self.metadata = MetaData()


    def saving_pre_processed_comments(self, comments):
        pre_process_comments = Table('pre_process_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            with conn.begin():
                stmt = sa.insert(pre_process_comments).values(comment=comments)
                conn.execute(stmt)
  
    
    def get_all_pre_processed_comments(self):
        pre_process_comments = Table('pre_process_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(pre_process_comments.c.id,pre_process_comments.c.comment)
            result = conn.execute(stmt)
            return result.fetchall()
        
    def bulk_update_lemmatize(self, bulk_update_lemmatize_comment,website):
        pre_process_comments_table = Table('pre_process_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            with conn.begin():  # Begin a transaction
                for comment_id, reviews in bulk_update_lemmatize_comment:
                    stmt = sa.insert(pre_process_comments_table).values(id = int(comment_id), comment = reviews,website = website)
                    conn.execute(stmt)
                    #logging.debug(f"insert {int(comment_id)} with {reviews}")
            
