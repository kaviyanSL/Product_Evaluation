from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import pickle
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ClassificationModelService():

    def DNN_Classifier(self, data, target):
        # Ensure TensorFlow uses GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                logging.info("GPU is available and will be used for training.")
            except RuntimeError as e:
                logging.error(f"Error setting up GPU: {e}")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
        
        # Define the model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        
        # Predict on the test data
        y_pred = np.argmax(model.predict(X_test), axis=1)
        
        # Calculate and log the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model Accuracy: {accuracy}")
        
        # Generate and log the classification report
        class_report = classification_report(y_test, y_pred)
        logging.info(f"Classification Report:\n{class_report}")
        
        # Generate and log the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Serialize the model using pickle
        model_pickle = pickle.dumps(model)
        
        return model_pickle