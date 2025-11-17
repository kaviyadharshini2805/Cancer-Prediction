# 🩺 KNN Cancer Prediction

KNN Cancer Prediction is an interactive machine learning web application that predicts whether a patient has benign (0) or malignant (1) cancer based on structured medical and lifestyle attributes. Built using Python, scikit-learn, and Streamlit, it provides a user-friendly interface for real-time cancer risk prediction.

## 🚀 Features

Predicts cancer based on key patient attributes:

Age, Gender, BMI

Smoking status, Genetic risk

Physical activity, Alcohol intake

Family cancer history

Interactive Streamlit interface for real-time data input.

Scaled input features ensure model consistency with training data.

Lightweight KNN model ideal for small to medium structured datasets.

Clear output indicating Benign or Malignant prediction.

## 📊 Dataset

File: cancer_data.csv

Contains patient medical and lifestyle data with the target column diagnosis.

Sample format:

age	gender	bmi	smoking	genetic_risk	physical_activity	alcohol_intake	cancer_history	diagnosis
58	1	16.08	0	1	8.14	4.14	1	1
71	0	30.82	0	1	9.36	3.51	0	0

## ⚙ Installation

### Clone the repository:

git clone https://github.com/your-username/knn-cancer-prediction.git
cd knn-cancer-prediction


### Install dependencies:

pip install -r requirements.txt


### Train the model

python train_model.py


### Run the Streamlit app:

streamlit run app.py

## 🖥 Usage

Open the Streamlit app in your browser.

Enter patient data in the input fields.

Click Predict.

View the prediction:

✅ Benign (0)

⚠️ Malignant (1)

## 🧠 How It Works

Data Loading: Reads CSV containing patient features and diagnosis.

Data Splitting: Training and testing sets.

Feature Scaling: StandardScaler normalizes inputs.

Model Training: K-Nearest Neighbors learns patterns from the training data.

Model Saving: Saves KNN model, scaler, and feature names.

Streamlit Prediction: Scales user inputs and predicts cancer risk in real-time.

## 🗂 Project Structure
knn-cancer-prediction/

│

├── README.md                # Project overview and instructions

├── requirements.txt         # Python dependencies

├── cancer_data.csv          # Dataset with patient attributes & diagnosis

├── train_model.py           # Script to train KNN model

├── app.py                   # Streamlit web app for prediction

├── knn_cancer_model.pkl     # Trained KNN model

├── knn_scaler.pkl           # StandardScaler used in training

├── feature_names.pkl        # List of training feature names
