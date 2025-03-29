from django.core.management.base import BaseCommand
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

from datetime import datetime
from mood_prediction_np.models import BERTModel
from django.utils import timezone
from django.conf import settings
from tqdm import tqdm
import time
from torch.cuda.amp import autocast, GradScaler

# Paths for model storage
MODELS_DIR = os.path.join(settings.BASE_DIR, 'bert_model', 'en_model')
os.makedirs(MODELS_DIR, exist_ok=True)


class EnglishSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])  # Ensure labels are integers

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
            'targets': torch.tensor(label, dtype=torch.long)
        }


class SentimentClassifier_en(torch.nn.Module):
    def __init__(self, model_name, num_classes=6):
        super(SentimentClassifier_en, self).__init__()
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
    help = 'Train the BERT model for English sentiment analysis'

    def handle(self, *args, **kwargs):
        model_name = "bert-base-uncased"
        max_len = 128
        batch_size = 16
        epochs = 5
        learning_rate = 2e-5
        train_ratio = 0.8

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Load and clean dataset
        cleaned_data_path = os.path.join(
            settings.BASE_DIR, 'dataset', 'en_data_cleaned.csv')
        if not os.path.exists(cleaned_data_path):
            print("Cleaning data... Running clean_en_data management command.")
            from django.core.management import call_command
            call_command('clean_en_data')

        df = pd.read_csv(cleaned_data_path, encoding='utf-8')
        texts = df['text'].values
        labels = df['label'].astype(int).values  # Ensure labels are integers

        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=1 - train_ratio, random_state=42, stratify=labels
        )

        print(
            f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")

        # Tokenizer and datasets
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = EnglishSentimentDataset(
            X_train, y_train, tokenizer, max_len)
        val_dataset = EnglishSentimentDataset(X_val, y_val, tokenizer, max_len)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Model, optimizer, scheduler
        model = SentimentClassifier_en(model_name).to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = torch.nn.CrossEntropyLoss()

        scaler = GradScaler()  # For mixed precision training
        best_accuracy = 0
        model_save_path = None

        total_training_time = time.time()

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            train_losses = []

            print(f"\nEpoch {epoch + 1}/{epochs}")
            progress_bar = tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                targets = batch['targets'].to(device)

                optimizer.zero_grad()

                with autocast():  # Mixed precision training
                    outputs = model(input_ids, attention_mask, token_type_ids)
                    loss = loss_fn(outputs, targets)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_losses.append(loss.item())
                progress_bar.set_postfix(loss=loss.item())

            # Validation
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    targets = batch['targets'].to(device)

                    outputs = model(input_ids, attention_mask, token_type_ids)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()

                    all_preds.extend(preds)
                    all_labels.extend(targets.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')

            print(
                f"Epoch {epoch+1}: Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(MODELS_DIR, f"model_{timestamp}.pt")
                torch.save(model.state_dict(), model_path)
                tokenizer.save_pretrained(MODELS_DIR)
                model_save_path = model_path
                print(f"Model saved to {model_save_path}")

        print(
            f"\nTotal training time: {time.time() - total_training_time:.2f} seconds")
