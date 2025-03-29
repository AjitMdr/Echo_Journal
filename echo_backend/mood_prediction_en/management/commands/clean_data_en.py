from django.core.management.base import BaseCommand
import pandas as pd
import os
from django.conf import settings


class Command(BaseCommand):
    help = "Clean the English sentiment data"

    def handle(self, *args, **kwargs):
        """Clean the English sentiment data and save to cleaned file."""
        data_path = os.path.join(settings.BASE_DIR, 'dataset', 'en_data.csv')

        if not os.path.exists(data_path):
            self.stdout.write(self.style.ERROR(
                f"Data file '{data_path}' does not exist."))
            return

        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except Exception as e:
            self.stdout.write(self.style.ERROR(
                f"Error reading the CSV file: {str(e)}"))
            return

        # Ensure required columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            self.stdout.write(self.style.ERROR(
                "CSV format is incorrect. Required columns: 'text', 'label'."))
            return

        # Remove empty rows
        initial_count = df.shape[0]
        df = df.dropna(subset=['text', 'label'])
        df = df[df['text'].str.strip() != '']
        cleaned_count = df.shape[0]

        # Convert labels to integers and filter valid classes
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df[df['label'].isin([0, 1, 2, 3, 4, 5])]

        # Save cleaned data
        cleaned_data_path = os.path.join(
            settings.BASE_DIR, 'dataset', 'en_data_cleaned.csv')
        df.to_csv(cleaned_data_path, index=False, encoding='utf-8')

        self.stdout.write(self.style.SUCCESS(
            f"Data cleaned: {initial_count - cleaned_count} rows removed"))
        self.stdout.write(self.style.SUCCESS(
            f"Cleaned data saved to '{cleaned_data_path}'"))
