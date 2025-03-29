# mood_prediction_np/management/commands/train_bert_model.py
from django.core.management.base import BaseCommand
import pandas as pd
import os
from django.conf import settings
from mood_prediction_np.utils import clean_nepali_text  # Import the utility function

class Command(BaseCommand):
    help = "Clean the Nepali sentiment data"

    def handle(self, *args, **kwargs):
        """Clean the Nepali sentiment data and save to cleaned file."""
        # Get path to the data file
        data_path = os.path.join(settings.BASE_DIR, 'dataset', 'np_data.csv')
        
        if not os.path.exists(data_path):
            self.stdout.write(self.style.ERROR(f"Data file '{data_path}' does not exist."))
            return

        # Read the CSV file
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error reading the CSV file: {str(e)}"))
            return
        
        # Check if columns exist, if not create appropriate structure
        if 'Sentences' not in df.columns or 'Sentiment' not in df.columns:
            # If file format is just index,text,sentiment without headers
            if df.shape[1] == 3:
                df.columns = ['Index', 'Sentences', 'Sentiment']
            else:
                raise ValueError("CSV format is not as expected. Please check the file structure.")
        
        # Clean the text data
        df['clean_text'] = df['Sentences'].apply(clean_nepali_text)  # Use the utility function
        
        # Remove empty rows
        initial_count = df.shape[0]
        df = df[df['clean_text'].str.strip() != '']
        cleaned_count = df.shape[0]
        
        # Convert sentiment labels to integers
        df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')
        
        # Drop rows with NaN sentiment values
        df = df.dropna(subset=['Sentiment'])
        
        # Ensure sentiment is -1, 0, or 1
        df = df[df['Sentiment'].isin([-1, 0, 1])]
        
        # Save the cleaned data
        cleaned_data_path = os.path.join(settings.BASE_DIR, 'dataset', 'np_data_cleaned.csv')
        df.to_csv(cleaned_data_path, index=False, encoding='utf-8')
        
        self.stdout.write(self.style.SUCCESS(f"Data cleaned: {initial_count - cleaned_count} rows removed"))
        self.stdout.write(self.style.SUCCESS(f"Cleaned data saved to '{cleaned_data_path}'"))
