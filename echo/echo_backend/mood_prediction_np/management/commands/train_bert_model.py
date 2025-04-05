from django.core.management.base import BaseCommand
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import torch 
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
import numpy as np
from datetime import datetime
from mood_prediction_np.models import BERTModel
from django.utils import timezone
from django.conf import settings
from tqdm import tqdm  
import time  

# Ensure models directory exists
MODELS_DIR = os.path.join(settings.BASE_DIR, 'bert_model', 'model')
os.makedirs(MODELS_DIR, exist_ok=True)
 
class NepaliSentimentDataset(Dataset):
    def __init__(self, texts, sentiments, tokenizer, max_len):
        self.texts = texts
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sentiment = self.sentiments[idx]

        sentiment_mapped = sentiment + 1  # Convert from [-1, 0, 1] to [0, 1, 2]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'targets': torch.tensor(sentiment_mapped, dtype=torch.long)
        }

class SentimentClassifier(torch.nn.Module):
    def __init__(self, model_name, num_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

class Command(BaseCommand):
    help = 'Train the BERT model for Nepali sentiment analysis'

    def handle(self, *args, **kwargs):
        model_name = "bert-base-multilingual-cased"
        max_len = 128
        batch_size = 16
        epochs = 5
        learning_rate = 2e-5
        train_ratio = 0.8

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Load dataset
        cleaned_data_path = os.path.join(settings.BASE_DIR, 'dataset', 'np_data_cleaned.csv')
        if not os.path.exists(cleaned_data_path):
            print("Cleaning data... Running clean_data management command.")
            from django.core.management import call_command
            call_command('clean_data')

        df = pd.read_csv(cleaned_data_path, encoding='utf-8')
        texts = df['clean_text'].values
        sentiments = df['Sentiment'].values

        X_train, X_val, y_train, y_val = train_test_split(
            texts, sentiments, test_size=1 - train_ratio, random_state=42, stratify=sentiments
        )

        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = NepaliSentimentDataset(X_train, y_train, tokenizer, max_len)
        val_dataset = NepaliSentimentDataset(X_val, y_val, tokenizer, max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = SentimentClassifier(model_name).to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_accuracy = 0
        model_save_path = None

        total_training_time = time.time()  # Start total training timer

        for epoch in range(epochs):
            start_time = time.time()  # Start timer for the epoch
            model.train()
            train_losses = []
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                targets = batch['targets'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                progress_bar.set_postfix(loss=loss.item())

            # Validation
            model.eval()
            val_losses, predictions, actual_labels = [], [], []
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc="Validating", leave=False)
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    targets = batch['targets'].to(device)

                    outputs = model(input_ids, attention_mask, token_type_ids)
                    loss = loss_fn(outputs, targets)

                    val_losses.append(loss.item())

                    _, preds = torch.max(outputs, dim=1)
                    predictions.extend(preds.cpu().tolist())
                    actual_labels.extend(targets.cpu().tolist())

            predictions_mapped = [p - 1 for p in predictions]
            actual_labels_mapped = [a - 1 for a in actual_labels]

            accuracy = accuracy_score(actual_labels_mapped, predictions_mapped)
            f1 = f1_score(actual_labels_mapped, predictions_mapped, average='weighted')

            epoch_time = time.time() - start_time  # Calculate epoch duration

            print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f} seconds")
            print(f"Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
            print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = os.path.join(MODELS_DIR, f"model_{timestamp}")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "model.pt")
                torch.save(model.state_dict(), model_path)
                tokenizer.save_pretrained(model_dir)
                model_save_path = model_path
                print(f"Model saved to {model_save_path}")

        total_training_time = time.time() - total_training_time  # Calculate total training time
        print(f"\nTotal training time: {total_training_time:.2f} seconds")

        if model_save_path:
            BERTModel.objects.filter(is_active=True).update(is_active=False)
            bert_model = BERTModel(
                name=f"BERT_Nepali_Sentiment_{timestamp}",
                description=f"Trained on {len(X_train)} samples, {epochs} epochs",
                model_path=model_save_path,
                accuracy=best_accuracy,
                f1_score=f1,
                training_completed=timezone.now(),
                is_active=True
            )
            bert_model.save()
            print(f"Model information saved to database with ID {bert_model.id}")
