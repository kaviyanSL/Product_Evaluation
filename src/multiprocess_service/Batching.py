from src.database.RawCommentRepository import RawCommentRepository
import pandas as pd
import os

class Batching:
    def __init__(self):
        self.RawCommentRepository = RawCommentRepository()

    def Batchsize(self):
        total_records = self.RawCommentRepository.get_number_of_raws()
        number_of_batches = os.cpu_count() - 4
        number_of_records_per_batch = total_records // number_of_batches
        return total_records, number_of_records_per_batch

    def call_batchsized_data(self, from_batchsized, to_batchsized):
        return self.RawCommentRepository.get_batchsized_data(from_batchsized, to_batchsized)