
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

# main.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scripts.evaluation import evaluate_model

# Define Hugging Face model path
HUGGINGFACE_MODEL_PATH = "your-username/sentiment-analysis-model"

def evaluate_pipeline():
    print("Loading Hugging Face model for evaluation...")
    # Load the Hugging Face model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_PATH)

    # Path to preprocessed data for evaluation
    PREPROCESSED_DATA_PATH = "path_to_your_preprocessed_data"  # Update this to your actual file path

    print("Starting model evaluation...")
    # Evaluate the model
    evaluate_model(model, tokenizer, PREPROCESSED_DATA_PATH)
    print("Model evaluation completed!")


def main():
    print("Welcome to the Sentiment Analysis Testing!")
    print("Choose an operation to perform:")

    print("1: Evaluate the model")
    print("2: Exit")

    try:
        # Prompt user for a choice
        choice = int(input("Enter your choice (1/2): ").strip())

        if choice == 1:
            evaluate_pipeline()

        elif choice == 2:
            print("Exiting. Goodbye!")

        else:
            print("Invalid choice. Please choose between 1 and 2.")

    except ValueError:
        print("Invalid input. Please enter a number (1/2).")


if __name__ == "__main__":
    main()
