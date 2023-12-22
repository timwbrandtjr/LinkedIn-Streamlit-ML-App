import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Part 1
s = pd.read_csv("social_media_usage.csv")

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

def kpi_box(title, value):
    return f"""
        <div style="
            display: inline-block;
            min-width: 200px;  
            margin-right: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            ">
            <h3 style="font-size: 24px;">{title}</h3>
            <p style="font-size: 18px;">{value}</p>
        </div>
    """
    
def main():
    st.title("LinkedIn User Prediction App")

    st.markdown("""
            For this app, a logistic regression model was trained to predict LinkedIn usage based on demographic and social attributes. 
            Features including income, education, parental and marital status, gender, and age were carefully selected for their relevance 
            to social media habits. The model was trained on an 80/20 train/test split, and logistic regression was chosen specifically for its 
            interpretability. Performance metrics like accuracy, precision, recall and F1 score offered insights into feature importance to enhance 
            real-time predictions. Ideally, the Decision Science team could leverage such a model to facilitate meaningful discussions within the marketing 
            team about campaign options for targeting customer segments. The app provides an intuitive interface for exploring those options.
            """)

    # Sidebar with user input
    st.sidebar.header("User Input Features")
    income = st.sidebar.slider("Income", 1, 9, 5, key="income_slider")
    education = st.sidebar.slider("Education", 1, 8, 4, key="education_slider")
    parent = st.sidebar.radio("Parent", ["No", "Yes"], key="parent_radio", help="Parental Status", )
    marital_status = st.sidebar.radio("Marital Status", ["Single", "Married"], key="marital_status_radio")
    gender = st.sidebar.radio("Gender", ["Male", "Female"], key="gender_radio")
    age = st.sidebar.slider("Age", 18, 98, 30, key="age_slider")

    # Load the fitted scaler and model
    scaler = load_scaler()
    model = load_model()

    # Create a DataFrame with all features (including user input)
    all_features = pd.DataFrame({
        'income': [income],
        'educ2': [education],
        'par': [1 if parent == "Yes" else 0],
        'marital': [1 if marital_status == "Married" else 0],
        'gender': [1 if gender == "Female" else 0],
        'age': [age]
    }, columns=X.columns)  # Use X.columns instead of X_train.columns

    # Make predictions
    probability = predict_probability(all_features, scaler, model)
    probability_pct = probability[0] * 100

# Display prediction results
    st.write("## Prediction")
    col1, col2 = st.columns(2)
    col1.markdown(kpi_box("Probability of being a LinkedIn user", f"{probability_pct:.2f}%"), unsafe_allow_html=True)

    if probability_pct >= 50:
        prediction = "LinkedIn User"
    else:
        prediction = "Non-LinkedIn User"

    col2.markdown(kpi_box("Prediction", prediction), unsafe_allow_html=True)


if __name__ == '__main__':
    main()

