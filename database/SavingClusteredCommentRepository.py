import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData

class SavingClusteredCommentRepository:
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

    def saving_raw_comments(self, comments):
        raw_comments = Table('raw_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.insert(raw_comments).values(comment=comments)
            conn.execute(stmt)
            conn.commit()
            
