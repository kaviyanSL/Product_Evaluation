import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData
from src.database.db_connection import DBConnection


class ClusteredCommentRepository:
    def __init__(self):
        self.db_connection = DBConnection()
        self.engine = self.db_connection.get_engine()
        self.metadata = MetaData()

    def save_clustered_comments(self, comment, cluster, vectorize_comment = None):
        clustered_comment = Table('clustered_comment', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            with conn.begin():
                stmt = sa.insert(clustered_comment).values(comment=comment, cluster=cluster, 
                                                        vectorize_comment=vectorize_comment)
                conn.execute(stmt)

    def get_all_clustered_comments(self):
        clustered_comment_table = Table('clustered_comment', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(clustered_comment_table)
            result = conn.execute(stmt)
            return result.fetchall()
            
