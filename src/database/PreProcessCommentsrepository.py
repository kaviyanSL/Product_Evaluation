import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData

class PreProcessCommentsrepository:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://root:root@localhost:3307/product_db")
        self.metadata = MetaData()


    def saving_pre_processed_comments(self, comments):
        pre_process_comments = Table('pre_process_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.insert(pre_process_comments).values(comment=comments)
            conn.execute(stmt)
            conn.commit()
    
    def get_all_pre_processed_comments(self):
        pre_process_comments = Table('pre_process_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(pre_process_comments.c.comment)
            result = conn.execute(stmt)
            return result.fetchall()
            
