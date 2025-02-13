import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData
from src.database.db_connection import DBConnection

class RawCommentRepository:
    def __init__(self):
        self.db_connection = DBConnection()
        self.engine = self.db_connection.get_engine()
        self.metadata = MetaData()

    def saving_raw_comments(self, comments, lang):
        raw_comments = Table('raw_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.insert(raw_comments).values(comment=comments, language=lang)
            conn.execute(stmt)
            conn.commit()

    def get_all_raw_comments(self):
        raw_comments_table = Table('raw_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(raw_comments_table.c.comment)
            result = conn.execute(stmt)
            return result.fetchall()

    def get_number_of_raws(self):
        raw_comments_table = Table('raw_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(sa.func.count(raw_comments_table.c.comment))
            result = conn.execute(stmt)
            return result.scalar()

    def get_batchsized_data(self, from_batchsized, to_batchsized):
        raw_comments_table = Table('raw_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(raw_comments_table).order_by(raw_comments_table.c.id).limit(
                to_batchsized - from_batchsized).offset(from_batchsized)
            result = conn.execute(stmt)
            return result.fetchall()

    def updating_language(self, comment_id, language):
        raw_comments_table = Table('raw_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.update(raw_comments_table).where(raw_comments_table.c.id == comment_id).values(language=language)
            conn.execute(stmt)
            conn.commit()