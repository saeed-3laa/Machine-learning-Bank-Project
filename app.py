import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Load the cleaned and encoded data
cleaned_data_path = "Cleaned_Bank.csv"
encoded_data_path = "Cleaned_encoded_Bank.csv"
model_path = "stacking_pipeline_model.pkl"

# Load data
cleaned_data = pd.read_csv(cleaned_data_path)
encoded_data = pd.read_csv(encoded_data_path)

# Apply SMOTE to balance the dataset
X = encoded_data.drop(columns=["y"])  # Assuming 'y' is the target column
y = encoded_data["y"]

# Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the balanced data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Load the trained model (this should include preprocessing steps like scaling)
model = joblib.load(model_path)

# Evaluate the model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit app
def main():
    # Set up the Streamlit page
    st.title("Machine Learning Model Deployment")

    # Data Overview Page
    st.header("Data Overview")
    st.subheader("Cleaned Data")
    st.write(cleaned_data.head())
    
    st.subheader("Statistical Summary")
    st.write(encoded_data.describe())

    # Plot correlation matrix without numbers
    plt.figure(figsize=(12, 8))
    corr_matrix = encoded_data.corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt='.2f', cbar=True)
    st.pyplot()

    # Visualize data before and after balancing (SMOTE)
    st.header("Target (y) Distribution Before and After SMOTE")
    
    # Original class distribution (before SMOTE)
    original_class_dist = y.value_counts()
    st.subheader("Class Distribution Before SMOTE")
    st.bar_chart(original_class_dist)

    # SMOTE class distribution (after SMOTE)
    smote_class_dist = y_res.value_counts()
    st.subheader("Class Distribution After SMOTE")
    st.bar_chart(smote_class_dist)

    # Model Report Page
    st.header("Model Report")
    st.text_area("Classification Report", report, height=300)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Prediction Page
    st.header("Class Prediction")
    st.write("Enter the features to predict the class:")

    # Create input fields for the original features (before encoding)
    features = cleaned_data.drop(columns=["y"]).columns.tolist()  # Use original features
    user_input = {}
    for feature in features:
        user_input[feature] = st.text_input(f"{feature}:")

    # Convert user inputs into a format the model can predict on
    if st.button("Predict"):
        try:
            # Convert user inputs into a DataFrame
            input_data = pd.DataFrame([user_input])
            
            # Perform encoding using get_dummies (same as in training)
            input_data_encoded = pd.get_dummies(input_data)
            
            # Align with the model's feature set
            input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)
            
            # Make prediction using the model (scaling is handled within the pipeline)
            prediction = model.predict(input_data_encoded)
            st.write(f"Prediction: {prediction[0]}")
        except ValueError:
            st.write("Please enter valid numerical values for all features.")



if __name__ == '__main__':
    main()
