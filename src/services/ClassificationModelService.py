from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import numpy as np
import pickle
import logging
import os
from datasets import Dataset
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ClassificationModelService():

    def DNN_Classifier(self, data, target):
        # Ensure TensorFlow uses GPU if available
        # physical_devices = tf.config.list_physical_devices('GPU')
        # if physical_devices:
        #     try:
        #         tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #         logging.info("GPU is available and will be used for training.")
        #     except RuntimeError as e:
        #         logging.error(f"Error setting up GPU: {e}")


        # Force TensorFlow to use CPU only
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    
    def bert_classifier(self, data, target):
        # Convert data to the correct format
        dataset = Dataset.from_dict({
            'input_ids': [d.tolist() for d in data],
            'labels': target.tolist()
        })

        # Split the data into training and testing sets
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
        # Load the BERT model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(target)))
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        # Train the model
        trainer.train()

        # Predict on the test data
        predictions = trainer.predict(test_dataset).predictions
        y_pred = np.argmax(predictions, axis=1)

        # Calculate and log the accuracy
        accuracy = accuracy_score(test_dataset['labels'], y_pred)
        logging.info(f"Model Accuracy: {accuracy}")

        # Generate and log the classification report
        class_report = classification_report(test_dataset['labels'], y_pred)
        logging.info(f"Classification Report:\n{class_report}")

        # Generate and log the confusion matrix
        conf_matrix = confusion_matrix(test_dataset['labels'], y_pred)
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Serialize the model using pickle
        model_pickle = pickle.dumps(model)
        
        return model_pickle