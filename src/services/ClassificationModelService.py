from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import scipy
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import numpy as np
import pickle
import logging
import os
from datasets import Dataset
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ClassificationModelService():

    def DNN_Classifier_old(self, data, target):
        # Ensure TensorFlow uses GPU if available
        # physical_devices = tf.config.list_physical_devices('GPU')
        # if physical_devices:
        #     try:
        #         tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #         logging.info("GPU is available and will be used for training.")
        #     except RuntimeError as e:
        #         logging.error(f"Error setting up GPU: {e}")



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
            'input_ids': data.toarray().tolist(),  # Convert sparse matrix to dense and then to list
            'labels': target.tolist()
        })

        # Split the data into training and testing sets
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
        # Define the model architecture
        model = torch.nn.Sequential(
            torch.nn.Linear(data.shape[1], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, len(np.unique(target))),
            torch.nn.Softmax(dim=1)
        )
        
        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Convert datasets to DataLoader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Training loop
        model.train()
        for epoch in range(3):  # Number of epochs
            running_loss = 0.0
            for inputs in train_loader:
                input_ids = torch.tensor(inputs['input_ids'])
                labels = torch.tensor(inputs['labels'])
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            logging.info(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        # Evaluation loop
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs in test_loader:
                input_ids = torch.tensor(inputs['input_ids'])
                labels = torch.tensor(inputs['labels'])
                
                outputs = model(input_ids)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate and log the accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        logging.info(f"Model Accuracy: {accuracy}")

        # Generate and log the classification report
        class_report = classification_report(all_labels, all_preds)
        logging.info(f"Classification Report:\n{class_report}")

        # Generate and log the confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Serialize the model using pickle
        model_pickle = pickle.dumps(model)
        
        return model_pickle

    def dnn_classifier(self, data, target):
        # Ensure TensorFlow uses GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                logging.info("GPU is available and will be used for training.")
            except RuntimeError as e:
                logging.error(f"Error setting up GPU: {e}")
        else:
            logging.info("GPU is not available. Using CPU.")

        # Convert sparse matrix to dense format
        data_dense = data.toarray()

        X_train, X_test, y_train, y_test = train_test_split(data_dense, target, test_size=0.2, random_state=42)

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
        
        # Define the callbacks
        callbacks = [
            tf.keras.callbacks.ProgbarLogger(count_mode='steps')
        ]
        
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)
        
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
    
    
    
    def bert_vector_classifier_v2(self, raw_texts, target):
        # Check if a GPU is available and use it
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Load pre-trained BERT tokenizer and model for sequence classification
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(np.unique(target)))
        model.to(device)  # Move the model to the GPU if available

        # Tokenize the raw texts
        encodings = tokenizer(raw_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

        # Convert target labels to tensor
        labels = torch.tensor(target, dtype=torch.long)

        # Split data into training and testing sets
        train_inputs, test_inputs, train_labels, test_labels = train_test_split(
            encodings["input_ids"], labels, test_size=0.2, random_state=42
        )

        # Create dataset objects
        train_dataset = Dataset.from_dict({"input_ids": train_inputs.tolist(), "labels": train_labels.tolist()})
        test_dataset = Dataset.from_dict({"input_ids": test_inputs.tolist(), "labels": test_labels.tolist()})

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            fp16=True,  # Enable mixed precision training
            gradient_accumulation_steps=2,  # Simulate larger batch size
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

        # Evaluate the model
        eval_result = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_result}")

        # Serialize the model using pickle
        model_pickle = pickle.dumps(model)
        logging.info(f"pickeling is done")

        return model_pickle