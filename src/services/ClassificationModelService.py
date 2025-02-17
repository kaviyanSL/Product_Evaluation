from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import numpy as np
import logging
import pickle
import os
from datasets import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ClassificationModelService:
    def __init__(self):
        # Set up GPU usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    def mlp_classifier(self, data, target):
        """Train an MLP model on GPU"""
        X_train, X_test, y_train, y_test = train_test_split(data.toarray(), target, test_size=0.2, random_state=42)

        # Convert data to tensors and move to GPU
        X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train, y_test = torch.tensor(y_train, dtype=torch.long).to(self.device), torch.tensor(y_test, dtype=torch.long).to(self.device)

        # Define MLP architecture
        model = torch.nn.Sequential(
            torch.nn.Linear(X_train.shape[1], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, len(np.unique(target))),
            torch.nn.Softmax(dim=1)
        ).to(self.device)

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).argmax(dim=1)

        accuracy = accuracy_score(y_test.cpu(), y_pred.cpu())
        logging.info(f"Model Accuracy: {accuracy}")

        # Serialize model
        model_pickle = pickle.dumps(model)
        return model_pickle

    def bert_classifier(self, raw_texts, target):
        """Train a BERT-based classifier on GPU"""
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(np.unique(target))).to(self.device)

        # Tokenize text and move tensors to GPU
        encodings = tokenizer(raw_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = encodings["input_ids"].to(self.device)
        labels = torch.tensor(target, dtype=torch.long).to(self.device)

        # Split data
        train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_dict({"input_ids": train_inputs.cpu().tolist(), "labels": train_labels.cpu().tolist()})
        test_dataset = Dataset.from_dict({"input_ids": test_inputs.cpu().tolist(), "labels": test_labels.cpu().tolist()})

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
        )

        # Train using Trainer API
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        trainer.train()

        # Evaluate model
        eval_result = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_result}")

        # Save model
        model_path = "bert_model.pth"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

        return model_path
