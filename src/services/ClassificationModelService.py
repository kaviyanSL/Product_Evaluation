from sklearn.neural_network import MLPClassifier
import pickle

class ClassificationModelService():

    def MLP_Classifier(self, X_train, y_train):
        clf = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', max_iter=10)
        clf_model = clf.fit(X_train, y_train)
        model_pickle = pickle.dumps(clf_model)
        return (model_pickle)
