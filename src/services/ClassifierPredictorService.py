import pickle
import torch
from transformers import BertForSequenceClassification


class ClassifierPredictorService:
    def __init__(self):
        pass

    def predict(self, model_pickle, data):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(np.unique(target)))
        model.load_state_dict(torch.load("/loaded_model.pth"))
        predictions = model.predict(data)
        return predictions