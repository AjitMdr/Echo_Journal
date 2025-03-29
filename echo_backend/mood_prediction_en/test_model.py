#!/usr/bin/env python
"""
Test the English BERT sentiment analysis model with custom text input.
Usage: python test_model.py "Your English text here"
"""

from . import predict
import django
import sys
import os

# Set up Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

# Import Django and set up
django.setup()

# Import directly from current directory


def main():
    """Run sentiment analysis on provided text or prompt user for input."""
    # Get input text
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = input("Enter English text for sentiment analysis: ")

    if not text.strip():
        print("Please provide some text for analysis.")
        return

    # Analyze sentiment
    print("\n----- Analysis Results -----")
    print(f"Text: {text}")

    try:
        # Get prediction from model or rule-based
        result = predict.predict_sentiment(text)

        # Display results
        print(f"\nPredicted Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(
            f"Method: {'Rule-based' if result.get('rule_based', False) else 'BERT model'}")

        # If rule-based, show matches
        if result.get('rule_based', False) and 'matches' in result:
            print("\nMatched emotion words:")
            for emotion, words in result['matches'].items():
                if words:
                    print(f"  {emotion}: {', '.join(words)}")

    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        print("\nTrying rule-based analysis only...")

        try:
            result = predict.predict_sentiment_rule_based(text)
            print(f"\nPredicted Sentiment (Rule-based): {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.2f}")

            if 'matches' in result:
                print("\nMatched emotion words:")
                for emotion, words in result['matches'].items():
                    if words:
                        print(f"  {emotion}: {', '.join(words)}")
        except Exception as e2:
            print(f"Rule-based analysis also failed: {e2}")


if __name__ == "__main__":
    main()
