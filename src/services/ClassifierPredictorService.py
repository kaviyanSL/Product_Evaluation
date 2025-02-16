import pickle

class ClassifierPredictorService:
    def __init__(self):
        pass

    def predict(self, model_pickle, data):
        model = pickle.loads(model_pickle)
        predictions = model.predict(data)
        return predictions