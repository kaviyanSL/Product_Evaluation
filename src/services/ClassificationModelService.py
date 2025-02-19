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

    def bert_classifier(self, raw_texts, target):
        os.environ["WANDB_DISABLED"] = "true"

        """Train a BERT-based classifier on GPU"""
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(np.unique(target))).to(self.device)

        # Tokenize text and move tensors to GPU
        logging.info(f"try to tokenize text")
        encodings = tokenizer(raw_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        logging.info(f"model tokenized")
        input_ids = encodings["input_ids"].to(self.device)
        labels = torch.tensor(target, dtype=torch.long).to(self.device)
        logging.info(f"try to train test spilit")
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
        logging.info(f"training is begined")
        trainer.train()
        logging.info(f"training is ended")

        # Evaluate model
        logging.info(f"try to eval trainer")
        eval_result = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_result}")

        # Save model
        model_path = "/models/bert_model.pth"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

        return model_path
