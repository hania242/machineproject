# evaluation.py
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def evaluate_model(model_path, preprocessed_data_path):
 
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(preprocessed_data_path, 'rb') as f:
        X_tfidf, y, _ = pickle.load(f)

  
    _, X_test, _, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

 
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

