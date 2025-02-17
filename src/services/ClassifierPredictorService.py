import numpy as np
import torch
from transformers import BertForSequenceClassification


class ClassifierPredictorService:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Loads the model from a file only once."""
        if self.model is None:
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()  # Set to evaluation mode
        return self.model

    def predict(self, data):
        """Runs model inference on given data."""
        model = self.load_model()
        with torch.no_grad():
            predictions = model(data)
        return np.argmax(predictions.logits.numpy(), axis=1)