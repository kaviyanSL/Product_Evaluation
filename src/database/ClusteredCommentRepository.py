import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData

class ClusteredCommentRepository:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://root:root@localhost:3307/product_db")
        self.metadata = MetaData()

    def save_clustered_comments(self, comment, cluster, vectorize_comment = None):
        clustered_comment = Table('clustered_comment', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.insert(clustered_comment).values(comment=comment, cluster=cluster, 
                                                       vectorize_comment=vectorize_comment)
            conn.execute(stmt)
            conn.commit()

    def get_all_clustered_comments(self):
        clustered_comment_table = Table('clustered_comment', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(clustered_comment_table)
            result = conn.execute(stmt)
            return result.fetchall()
            
