
# main.py
#from scripts.paths import RAW_DATA_PATH, CLEANSED_DATA_PATH, PREPROCESSED_DATA_PATH, RAW_TEXT_PATH, VOTING_MODEL_PATH
#from scripts.preprocessing import preprocess_data
#from scripts.training import train_model
#from scripts.evaluation import evaluate_model

# Preprocess data
#preprocess_data(RAW_DATA_PATH, CLEANSED_DATA_PATH, PREPROCESSED_DATA_PATH, RAW_TEXT_PATH)

#Train model
#model = train_model(PREPROCESSED_DATA_PATH, VOTING_MODEL_PATH)


# Evaluate model
#evaluate_model(VOTING_MODEL_PATH, PREPROCESSED_DATA_PATH)

# main.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scripts.paths import (
    RAW_DATA_PATH,
    CLEANSED_DATA_PATH,
    PREPROCESSED_DATA_PATH,
    RAW_TEXT_PATH,
)
from scripts.preprocessing import preprocess_data
from scripts.training import train_model
from scripts.evaluation import evaluate_model


# Define Hugging Face model path
HUGGINGFACE_MODEL_PATH = "your-username/sentiment-analysis-model"


# Preprocess, Train, and Evaluate functions using the Hugging Face model
def preprocess_pipeline():
    print("Starting data preprocessing...")
    preprocess_data(RAW_DATA_PATH, CLEANSED_DATA_PATH, PREPROCESSED_DATA_PATH, RAW_TEXT_PATH)
    print("Data preprocessing completed!")


def train_pipeline():
    print("Loading Hugging Face model for training...")
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_PATH)

    print("Starting model training...")
    train_model(PREPROCESSED_DATA_PATH, model, tokenizer)
    print("Model training completed!")


def evaluate_pipeline():
    print("Loading Hugging Face model for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_PATH)

    print("Starting model evaluation...")
    evaluate_model(model, tokenizer, PREPROCESSED_DATA_PATH)
    print("Model evaluation completed!")


def main():
    print("Welcome to the Sentiment Analysis Pipeline!")
    print("Choose an operation to perform:")

    print("1: Preprocess data")
    print("2: Train the model")
    print("3: Evaluate the model")
    print("4: Run complete pipeline (Preprocess, Train, Evaluate)")

    try:
        # Prompt user for a choice
        choice = int(input("Enter your choice (1/2/3/4): ").strip())

        if choice == 1:
            preprocess_pipeline()

        elif choice == 2:
            train_pipeline()

        elif choice == 3:
            evaluate_pipeline()

        elif choice == 4:
            preprocess_pipeline()
            train_pipeline()
            evaluate_pipeline()

        else:
            print("Invalid choice. Please choose between 1, 2, 3, and 4.")

    except ValueError:
        print("Invalid input. Please enter a number (1/2/3/4).")


if __name__ == "__main__":
    main()
