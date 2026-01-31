import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# Page Configuration
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training", "Model Evaluation", "Model Comparison", "Churn Prediction"])

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Customer Churn.csv")
    # Data Cleaning: Handle TotalCharges which often contains empty strings
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()

# Home Page
if page == "Home":
    st.title("📊 Customer Churn Analysis Project")
    st.markdown("""
    ### Business Problem Statement
    Customer churn occurs when customers stop doing business with a company. For telecommunication companies, 
    retaining existing customers is often more cost-effective than acquiring new ones. 
    This project aims to analyze customer behavior and develop machine learning models to predict 
    which customers are likely to churn, enabling proactive retention strategies.
    """)
    
    st.subheader("Dataset Overview")
    st.write(df.head())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean():.1%}")

# Data Exploration Page
elif page == "Data Exploration":
    st.title("🔍 Data Exploration (EDA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistical Summary")
        st.write(df.describe())
    
    with col2:
        st.subheader("Missing Values Check")
        st.write(df.isnull().sum())
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Churn Distribution (Count)")
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax, palette='viridis')
        st.pyplot(fig)
        
    with col4:
        st.subheader("Churn Distribution (%)")
        fig, ax = plt.subplots()
        df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=['#4CAF50', '#FF5252'])
        ax.set_ylabel('')
        st.pyplot(fig)

    st.divider()
    
    st.subheader("Feature Analysis vs Churn")
    col5, col6 = st.columns(2)
    
    with col5:
        st.write("**Gender vs Churn**")
        fig, ax = plt.subplots()
        sns.countplot(x='gender', hue='Churn', data=df, ax=ax)
        st.pyplot(fig)
        
    with col6:
        st.write("**Senior Citizen vs Churn**")
        fig, ax = plt.subplots()
        sns.countplot(x='SeniorCitizen', hue='Churn', data=df, ax=ax)
        st.pyplot(fig)
        
    col7, col8 = st.columns(2)
    
    with col7:
        st.write("**Contract Type vs Churn**")
        fig, ax = plt.subplots()
        sns.countplot(x='Contract', hue='Churn', data=df, ax=ax)
        st.pyplot(fig)
        
    with col8:
        st.write("**Payment Method vs Churn**")
        fig, ax = plt.subplots()
        sns.countplot(x='PaymentMethod', hue='Churn', data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Tenure Distribution by Churn")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(df[df['Churn'] == 'No']['tenure'], fill=True, label='No Churn', ax=ax)
    sns.kdeplot(df[df['Churn'] == 'Yes']['tenure'], fill=True, label='Churn', ax=ax)
    ax.legend()
    st.pyplot(fig)

# Model Training Page
elif page == "Model Training":
    st.title("⚙️ Model Training & Preprocessing")
    
    st.info("Applying Label Encoding and Standard Scaling to the dataset.")
    
    # Preprocessing
    df_ml = df.drop('customerID', axis=1).copy()
    
    # Encoding categorical variables
    le = LabelEncoder()
    categorical_cols = df_ml.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_ml[col] = le.fit_transform(df_ml[col])
        
    X = df_ml.drop('Churn', axis=1)
    y = df_ml['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.success("Data successfully preprocessed and split!")
    
    # Store data in session state for other pages
    st.session_state['X_train'] = X_train_scaled
    st.session_state['X_test'] = X_test_scaled
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    
    st.subheader("Train Machine Learning Models")
    if st.button("🚀 Start Training All Models"):
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "SVM": SVC(probability=True)
        }
        
        results = {}
        progress_bar = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "report": classification_report(y_test, y_pred, output_dict=True),
                "cm": confusion_matrix(y_test, y_pred),
                "y_prob": y_prob
            }
            progress_bar.progress((i + 1) / len(models))
        
        st.session_state['results'] = results
        st.session_state['trained_models'] = models
        st.success("All models trained successfully!")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.title("📈 Model Evaluation Metrics")
    
    if 'results' not in st.session_state:
        st.warning("Please train the models first on the 'Model Training' page.")
    else:
        model_name = st.selectbox("Select Model to Evaluate", list(st.session_state['results'].keys()))
        res = st.session_state['results'][model_name]
        y_test = st.session_state['y_test']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy Score", f"{res['accuracy']:.4f}")
            st.metric("ROC-AUC Score", f"{res['roc_auc']:.4f}")
            
            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(res['report']).transpose())
            
        with col2:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
        st.divider()
        
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {res['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend()
        st.pyplot(fig)

# Model Comparison Page
elif page == "Model Comparison":
    st.title("📊 Model Comparison & Selection")
    
    if 'results' not in st.session_state:
        st.warning("Please train the models first on the 'Model Training' page.")
    else:
        results = st.session_state['results']
        comparison_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": [res['accuracy'] for res in results.values()],
            "ROC-AUC": [res['roc_auc'] for res in results.values()]
        }).sort_values(by="Accuracy", ascending=False)
        
        st.subheader("Performance Comparison Table")
        st.table(comparison_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            fig, ax = plt.subplots()
            sns.barplot(x='Model', y='Accuracy', data=comparison_df, palette='magma', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col2:
            st.subheader("ROC-AUC Comparison")
            fig, ax = plt.subplots()
            sns.barplot(x='Model', y='ROC-AUC', data=comparison_df, palette='viridis', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Churn Prediction Page
elif page == "Churn Prediction":
    st.title("🔮 Real-Time Churn Prediction")
    
    if 'results' not in st.session_state or 'trained_models' not in st.session_state:
        st.warning("Please train the models first on the 'Model Training' page.")
    else:
        # Identify the best model based on Accuracy
        results = st.session_state['results']
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model = st.session_state['trained_models'][best_model_name]
        
        st.success(f"Using the best-performing model: **{best_model_name}**")
        
        # User Inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", df['gender'].unique())
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", df['Partner'].unique())
            Dependents = st.selectbox("Dependents", df['Dependents'].unique())
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

        with col2:
            st.subheader("Services")
            PhoneService = st.selectbox("Phone Service", df['PhoneService'].unique())
            MultipleLines = st.selectbox("Multiple Lines", df['MultipleLines'].unique())
            InternetService = st.selectbox("Internet Service", df['InternetService'].unique())
            OnlineSecurity = st.selectbox("Online Security", df['OnlineSecurity'].unique())
            OnlineBackup = st.selectbox("Online Backup", df['OnlineBackup'].unique())
            DeviceProtection = st.selectbox("Device Protection", df['DeviceProtection'].unique())
            TechSupport = st.selectbox("Tech Support", df['TechSupport'].unique())
            StreamingTV = st.selectbox("Streaming TV", df['StreamingTV'].unique())
            StreamingMovies = st.selectbox("Streaming Movies", df['StreamingMovies'].unique())

        with col3:
            st.subheader("Account")
            Contract = st.selectbox("Contract", df['Contract'].unique())
            PaperlessBilling = st.selectbox("Paperless Billing", df['PaperlessBilling'].unique())
            PaymentMethod = st.selectbox("Payment Method", df['PaymentMethod'].unique())
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=500.0)

        # Preprocessing Input Data
        input_data = pd.DataFrame({
            'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner], 
            'Dependents': [Dependents], 'tenure': [tenure], 'PhoneService': [PhoneService], 
            'MultipleLines': [MultipleLines], 'InternetService': [InternetService], 
            'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup], 
            'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport], 
            'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies], 
            'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling], 
            'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [MonthlyCharges], 
            'TotalCharges': [TotalCharges]
        })

        if st.button("🚀 Predict Churn"):
            # Prepare data for preprocessing (need original data for fitting encoders/scaler)
            df_ml_temp = df.drop(['customerID', 'Churn'], axis=1).copy()
            df_ml_temp = pd.concat([df_ml_temp, input_data], axis=0)
            
            # Label Encoding
            le = LabelEncoder()
            categorical_cols = df_ml_temp.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df_ml_temp[col] = le.fit_transform(df_ml_temp[col])
            
            # Scaling
            scaler = StandardScaler()
            X_encoded = df_ml_temp.iloc[:-1] # All except the last row
            input_encoded = df_ml_temp.iloc[-1:] # The last row
            
            scaler.fit(X_encoded)
            input_scaled = scaler.transform(input_encoded)
            
            # Prediction
            prediction = best_model.predict(input_scaled)[0]
            probability = best_model.predict_proba(input_scaled)[0][1]
            
            st.divider()
            if prediction == 1:
                st.error(f"### Result: Customer will churn")
            else:
                st.success(f"### Result: Customer will not churn")
                
            st.write(f"**Churn Probability:** {probability:.2%}")
            st.progress(probability)
