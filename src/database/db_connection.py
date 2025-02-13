from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class DBConnection:
    def __init__(self):
        self._engine = create_engine("mysql+pymysql://root:root@localhost:3307/product_db")
        self._Session = sessionmaker(bind=self._engine)

    def get_session(self):
        return self._Session()

    def get_engine(self):
        return self._engine