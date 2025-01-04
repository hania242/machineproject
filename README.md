 
# Sentiment Analysis Project

This project focuses on **sentiment analysis** using a combined dataset from IMDB and Twitter. The model is trained with an ensemble of **XGBoost**, **LightGBM**, and **Logistic Regression**, and a **Streamlit GUI** is provided for user interaction.



## Project Overview

### 1. **Data Collection and Preprocessing**
- **Datasets Used**:
  - [IMDB Sentiment Dataset](https://www.kaggle.com/)
  - [Twitter Sentiment Dataset](https://www.kaggle.com/)
- **Data Combination**:
  - Both datasets were merged into a single combined dataset on Kaggle.
- **Preprocessing**:
  - Cleaned and preprocessed the dataset to handle missing values, remove unnecessary characters, and standardize text formats.
  
---

### 2. **Model Training**
- **Data Splitting**:
  - Training data: **80%**
  - Testing data: **20%**
- **Modeling Techniques**:
  - Used an ensemble of:
    - **XGBoost**
    - **LightGBM**
    - **Logistic Regression**
  - Trained the model on **data chunks** to handle large datasets efficiently.
  - Experimented with multiple epochs but found no significant performance improvement, so the model was trained with only **one epoch**.
- **Performance**:
  - The trained model achieved satisfactory accuracy and precision.

---

### 3. **Streamlit GUI**
- A **Streamlit-based Graphical User Interface (GUI)** was developed to:
  - Allow users to test the model with their own data.
  - Provide a simple and interactive experience for sentiment prediction.

---
link to datasets :
processed: https://www.dropbox.com/scl/fo/5prr5vkphvftcww42m2xb/AISLL19_NGXePoNsoNc1uG4?rlkey=l3ne0ijpwvbvsv80h7m58cy2z&st=i6tmo8jo&dl=0
raw: https://www.dropbox.com/scl/fo/w0y9yy24cvt7hys0lwy3x/AHDrFUP24ibYTpP3M0aa5z4?rlkey=ogsetl5d5vhwnsycvt22q00ag&st=ajf4k2wj&dl=0
