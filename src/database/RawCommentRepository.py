import sqlalchemy as sa
from sqlalchemy import create_engine, Table, MetaData
from src.database.db_connection import DBConnection
import logging


class RawCommentRepository:
    def __init__(self):
        self.db_connection = DBConnection()
        self.engine = self.db_connection.get_engine()
        self.metadata = MetaData()

    def saving_raw_comments(self, comments, lang):
        raw_comments = Table('raw_comments', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            with conn.begin():  # Begin a transaction
                stmt = sa.insert(raw_comments).values(comment=comments, language=lang)
                conn.execute(stmt)

    def get_all_raw_comments(self):
        raw_comments_table = Table('raw_comments_amazon', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(raw_comments_table.c.text).where(
                            raw_comments_table.c.language == 'en').order_by(
                            raw_comments_table.c.id)


            result = conn.execute(stmt)
            return result.fetchall()

    def get_number_of_raws(self):
        raw_comments_table = Table('raw_comments_amazon', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(sa.func.count(raw_comments_table.c.text)).where(
                            raw_comments_table.c.language == None)
            result = conn.execute(stmt)
            return result.scalar()

    def get_batchsized_data(self, from_batchsized, to_batchsized):
        raw_comments_table = Table('raw_comments_amazon', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            stmt = sa.select(raw_comments_table).order_by(raw_comments_table.c.id).limit(
                to_batchsized - from_batchsized).offset(from_batchsized)
            result = conn.execute(stmt)
            return result.fetchall()

    def updating_language(self, comment_id, language):
        raw_comments_table = Table('raw_comments_amazon', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            with conn.begin():  # Begin a transaction
                stmt = sa.update(raw_comments_table).where(raw_comments_table.c.id == comment_id).values(language=language)
                conn.execute(stmt)

    def bulk_update_language(self, updates):
        raw_comments_table = Table('raw_comments_amazon', self.metadata, autoload_with=self.engine)
        with self.engine.connect() as conn:
            with conn.begin():  # Begin a transaction
                for comment_id, language in updates:
                    stmt = sa.update(raw_comments_table).where(raw_comments_table.c.id == comment_id).values(language=language)
                    conn.execute(stmt)
                    logging.debug(f"update {comment_id} with {language}")
