import torch
import numpy as np
import logging
from datasets import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ClassificationModelService:
    def __init__(self):
        # Set up GPU usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    def bert_classifier(self, raw_texts, target):
        """Train a BERT-based classifier on GPU"""
        
        # Subset for faster testing
        raw_texts = raw_texts[:100]
        target = target[:100]
        
        # Convert labels to integers
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target)  # Convert string labels to integers
        num_labels = len(np.unique(target))  # Get the number of unique labels

        # Initialize tokenizer and model
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(self.device)

        # Tokenize text and move tensors to GPU
        logging.info(f"Starting tokenization")
        encodings = tokenizer(raw_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        logging.info(f"Tokenization complete")
        
        input_ids = encodings["input_ids"].to(self.device)
        labels = torch.tensor(target, dtype=torch.long).to(self.device)

        # Split data into training and testing sets
        logging.info(f"Splitting data into train/test sets")
        train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_dict({"input_ids": train_inputs.cpu().tolist(), "labels": train_labels.cpu().tolist()})
        test_dataset = Dataset.from_dict({"input_ids": test_inputs.cpu().tolist(), "labels": test_labels.cpu().tolist()})

        # Define training arguments with optimizations
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,  # Increased batch size
            per_device_eval_batch_size=16,  # Increased eval batch size
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            fp16=True,  # Enable mixed precision training
            save_steps=500,  # Save model every 500 steps
            gradient_accumulation_steps=4,  # Accumulate gradients for simulating larger batch size
            dataloader_num_workers=4,  # Parallelize data loading
            logging_steps=100,  # Log every 100 steps
            evaluation_strategy='epoch',  # Evaluate at the end of each epoch
        )

        # Use AdamW optimizer for better convergence
        optimizer = AdamW(model.parameters(), lr=5e-5)

        # Train using Trainer API
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            optimizers=(optimizer, None),
        )

        logging.info(f"Training started")
        trainer.train()
        logging.info(f"Training finished")

        # Evaluate model
        logging.info(f"Evaluating model")
        eval_result = trainer.evaluate()
        logging.info(f"Evaluation results: {eval_result}")

        # Save model
        model_path = "./models/bert_model.pth"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

        # Clean up to free GPU and memory
        logging.info("Cleaning up to free memory...")
        
        torch.cuda.empty_cache()

        del model
        del trainer
        del train_dataset
        del test_dataset
        del encodings
        del input_ids
        del labels

        gc.collect()

        torch.cuda.empty_cache()

        logging.info("Memory cleaned up successfully.")

        return model_path
