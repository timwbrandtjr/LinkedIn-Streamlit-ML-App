import streamlit as st
import pandas as pd
import numpy as np

st.markdown("# Welcome to my streamlit app!")

# Q1 

s = pd.read_csv(r"C:\Users\timwb\OneDrive\Desktop\Georgetown MSBA\3. ProgrammingII.Data Infrastructure\Final Project\social_media_usage.csv")

# Q2
def clean_sm(x):
    return np.where(x==1, 1, 0)

# Q3
ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age', 'web1h']].copy()

# Update the gender column to be binary (1 for female, 0 for other)
ss['gender'] = np.where(ss['gender'] == 2, 1, 0)

# Apply clean_sm to the target column
ss['sm_li'] = ss['web1h'].apply(clean_sm)

# Drop missing values
ss = ss.dropna()

# Q4
y = ss['sm_li']
X = ss.drop('sm_li', axis=1)

# Q5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Q6
# Instantiate a logistic regression model with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42)

# Fit the model with the training data
model.fit(X_train, y_train)