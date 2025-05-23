# Bank Marketing Campaign Analysis and Predictive Modeling

## Project Overview
This project analyzes a bank marketing campaign dataset to predict whether a customer will subscribe to a term deposit. It includes exploratory data analysis (EDA), preprocessing, and predictive modeling using KNN, SVM, Decision Tree, Naive Bayes, and a stacking ensemble. The project also features a Streamlit app for model deployment.

## Files
- `bank_marketing_analysis.ipynb`: Jupyter Notebook with the full analysis and modeling.
- `stacking_pipeline_model.pkl`: Trained stacking model.
- `Cleaned_Bank.csv`: Cleaned dataset (not included in repository due to size).
- `Cleaned_encoded_Bank.csv`: Encoded dataset (not included in repository due to size).
- `app.py`: Streamlit app for model deployment.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imblearn, streamlit, joblib
- Install dependencies: `pip install -r requirements.txt`

## How to Run
1. Clone the repository: `git clone https://github.com/saeed-3laa/Machine-learning-Bank-Project.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Dataset
The dataset (`Bank.csv`) is not included due to its size.
## Results
- **Best Model**: Stacking ensemble (95% accuracy).
- See the notebook for detailed performance metrics and visualizations.
