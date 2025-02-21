import numpy as np
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer



class ClassifierPredictorService:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Loads the model from a file only once."""
        if self.model is None:
            # TODO: remember that the num_labels is set as a hardcode and it should be dynamic since we might have deferent num_labels
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()  # Set to evaluation mode
        return self.model

    def predict(self, data):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
        model = self.load_model()
        with torch.no_grad():
            predictions = model(inputs["input_ids"])
        return np.argmax(predictions.logits.numpy(), axis=1)
    
    
    def predict_list(self, data):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
        model = self.load_model()
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"])
        
        predictions = np.argmax(outputs.logits.numpy(), axis=1)
        
        unique, counts = np.unique(predictions, return_counts=True)
        return dict(zip(unique, counts))  
