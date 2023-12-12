import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Part 1
s = pd.read_csv(r"C:\Users\timwb\OneDrive\Desktop\Georgetown MSBA\3. ProgrammingII.Data Infrastructure\Final Project\social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age', 'web1h']].copy()

# Handle the 'income' column as ordered numeric from 1 to 9, above 9 considered missing
ss['income'] = np.where((ss['income'] >= 1) & (ss['income'] <= 9), ss['income'], np.nan)

# Handle the 'education' column as ordered numeric from 1 to 8, above 8 considered missing
ss['educ2'] = np.where((ss['educ2'] >= 1) & (ss['educ2'] <= 8), ss['educ2'], np.nan)

# Apply clean_sm to the 'par' column
ss['par'] = np.where(ss['par'] == 1, 0, 1)

# Apply clean function to marital
ss['marital'] = clean_sm(ss['marital'])

# Apply clean_sm to the 'gender' column
ss['gender'] = clean_sm(ss['gender'])

# Handle the 'age' column as numeric, above 98 considered missing
ss['age'] = np.where(ss['age'] <= 98, ss['age'], np.nan)

# Apply clean_sm to the target column
ss['sm_li'] = ss['web1h'].apply(clean_sm)

# Drop missing values
ss = ss.dropna()

# Create target vector (y) and feature set (X)
y = ss['sm_li']  # Target Vector
X = ss.drop(['sm_li', 'web1h'], axis=1)

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression model
logistic_model = LogisticRegression(class_weight="balanced", max_iter=1000)

# Fit model
logistic_model.fit(X_train_scaled, y_train)

# Part 2

def load_model():
    # Load your trained model here
    # For example: model = joblib.load('your_model_file.pkl')
    return logistic_model

def load_scaler():
    # Load your fitted scaler here
    # For example: scaler = joblib.load('your_scaler_file.pkl')
    return scaler

def predict_probability(features, scaler, model):
    # Standardize features
    features_scaled = scaler.transform(features)
    # Make prediction
    probability = model.predict_proba(features_scaled)[:, 1]
    return probability

def main():
    st.title("LinkedIn User Prediction App")

    # Sidebar with user input
    st.sidebar.header("User Input Features")
    income = st.sidebar.slider("Income", 1, 9, 5)
    education = st.sidebar.slider("Education", 1, 8, 4)
    parent = st.sidebar.radio("Parent", ["No", "Yes"])
    marital_status = st.sidebar.radio("Marital Status", ["Single", "Married"])
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 98, 30)

    # Load the fitted scaler
    scaler = load_scaler()

    # Display the user input features
    st.write("## User Input Features")
    user_input = pd.DataFrame({'income': [income], 'educ2': [education], 'par': [1 if parent == "Yes" else 0],
                               'marital': [1 if marital_status == "Married" else 0],
                               'gender': [1 if gender == "Female" else 0], 'age': [age]})
    st.table(user_input)

    # Load the model and make predictions
    model = load_model()
    probability = predict_probability(user_input, scaler, model)

    # Display prediction results
    st.write("## Prediction")
    st.write(f"Probability of being a LinkedIn user: {probability[0]:.2f}")

    prediction = "LinkedIn User" if probability >= 0.5 else "Non-LinkedIn User"
    st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()