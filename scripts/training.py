import pickle
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from scipy.sparse import issparse

def train_model(preprocessed_data_path, model_path, chunk_size=1000):
    
    with open(preprocessed_data_path, 'rb') as f:
        X_tfidf, y, _ = pickle.load(f)


    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    
    log_reg = LogisticRegression(max_iter=500, random_state=42)
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    lgbm_model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    
    voting_model = VotingClassifier(
        estimators=[
            ('log_reg', log_reg),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        voting='soft'
    )

    
    total_samples = X_train.shape[0]
    indices = np.arange(total_samples)
    np.random.shuffle(indices)  

    with tqdm(total=total_samples, desc="Training Progress", leave=True) as pbar:
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_indices = indices[start_idx:end_idx]

           
            if issparse(X_train):
                X_chunk = X_train[chunk_indices].toarray() 
            else:
                X_chunk = X_train[chunk_indices]

        
            y_chunk = y_train.iloc[chunk_indices]

           
            voting_model.fit(X_chunk, y_chunk)

           
            pbar.update(end_idx - start_idx)


    with open(model_path, 'wb') as f:
        pickle.dump(voting_model, f)

    print(f"Model saved to {model_path}")

   
    return voting_model
