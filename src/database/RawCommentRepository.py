import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData

class RawCommentRepository:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://root:root@localhost:3307/product_db")
        self.metadata = MetaData()


    def saving_raw_comments(self, comments,lang):
        raw_comments = Table('raw_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.insert(raw_comments).values(comment=comments,language=lang)
            conn.execute(stmt)
            conn.commit()
    
    def get_all_raw_comments(self):
        raw_comments_table = Table('raw_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(raw_comments_table.c.comment)
            result = conn.execute(stmt)
            return result.fetchall()
            
