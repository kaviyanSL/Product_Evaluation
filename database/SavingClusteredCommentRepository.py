import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData

class SavingClusteredCommentRepository:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://root:root@localhost:3307/product_db")
        self.metadata = MetaData()
        self.clustered_comment = Table('clustered_comment', self.metadata, autoload_with=self.engine)

    def save_clustered_comments(self, comment, clustered_comment):
        with self.engine.connect() as conn:
            stmt = sa.insert(self.clustered_comment).values(comment=comment, cluster=clustered_comment)
            conn.execute(stmt)
            conn.commit()
