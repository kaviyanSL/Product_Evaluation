from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import numpy as np
import logging
import gc
from datasets import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ClassificationModelService:
    def __init__(self):
        # Set up GPU usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    def bert_classifier(self, raw_texts, target):
        """Train a BERT-based classifier on GPU"""

        # Convert labels to integers
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target)  
        num_labels = len(np.unique(target))

        # Initialize tokenizer and model
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(self.device)

        # Tokenize text and move tensors to GPU
        logging.info("Starting tokenization")
        encodings = tokenizer(raw_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        logging.info("Tokenization complete")

        input_ids = encodings["input_ids"]
        labels = torch.tensor(target, dtype=torch.long)

        # Split data into training and testing sets
        logging.info("Splitting data into train/test sets")
        train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_dict({"input_ids": train_inputs.tolist(), "labels": train_labels.tolist()})
        test_dataset = Dataset.from_dict({"input_ids": test_inputs.tolist(), "labels": test_labels.tolist()})

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,  # Increase batch size if GPU allows
            per_device_eval_batch_size=16,   # Increase batch size for faster eval
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            fp16=True,  # Enable mixed precision training
            dataloader_num_workers=0,  # Fix multiprocessing issue on Windows
        )

        # Train using Trainer API
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        logging.info("Training started")
        trainer.train()
        logging.info("Training finished")

        # Evaluate model
        logging.info("Evaluating model")
        eval_result = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_result}")

        # Save model
        model_path = "./models/bert_model.pth"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

        # Free GPU memory after training
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return model_path
