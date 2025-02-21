import logging
import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData
from src.database.db_connection import DBConnection


class ClusteredCommentRepository:
    def __init__(self):
        self.db_connection = DBConnection()
        self.engine = self.db_connection.get_engine()
        self.metadata = MetaData()

    def save_clustered_comments(self, comments):
        clustered_comments_table = Table('clustered_comment', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            with conn.begin():  # Begin a transaction
                stmt = sa.insert(clustered_comments_table).values(comments)
                conn.execute(stmt)
                logging.debug("Inserted clustered comments")


    def get_all_clustered_comments(self):
        clustered_comment_table = Table('clustered_comment', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(clustered_comment_table)
            result = conn.execute(stmt)
            return result.fetchall()
        
    def get_specific_comment_data(self,row_id):
        clustered_comment_table = Table('clustered_comment', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(clustered_comment_table).where(clustered_comment_table.c.id == row_id)
            result = conn.execute(stmt)
            return result.fetchone()
            

    def get_all_clustered_comments_200(self):
        clustered_comment_table = Table('clustered_comment', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(clustered_comment_table).limit(2000)
            result = conn.execute(stmt)
            return result.fetchall()

