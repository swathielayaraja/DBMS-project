import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from scipy import stats

# Streamlit settings
st.title("EEG Data Analysis and Prediction App")
st.sidebar.header("EEG Dataset Analysis")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload EEG Dataset CSV", type="csv")
if uploaded_file:
    # Load data from uploaded file
    pf_1 = pd.read_csv(uploaded_file)

    # Dataset overview
    st.write("### Dataset Overview")
    st.write(pf_1.head())

    # Sidebar options
    if st.sidebar.checkbox("Show Dataset Info"):
        st.write("### Dataset Info")
        st.write(pf_1.info())

    if st.sidebar.checkbox("Show Descriptive Statistics"):
        st.write("### Descriptive Statistics")
        st.write(pf_1.describe())

    if st.sidebar.checkbox("Show Missing Values"):
        st.write("### Missing Values")
        st.write(pf_1.isnull().sum())

    if st.sidebar.checkbox("Show Data Duplicates"):
        st.write("### Data Duplicates")
        st.write(pf_1.duplicated().sum())

    # Histogram and Boxplot for each numerical column
    numerical_columns = pf_1.select_dtypes(include=np.number).columns
    if st.sidebar.checkbox("Show Histograms"):
        for col in numerical_columns:
            plt.hist(pf_1[pf_1['eyeDetection'] == 1][col], color='red', alpha=0.5, label='Eye Movement Detected')
            plt.hist(pf_1[pf_1['eyeDetection'] == 0][col], color='blue', alpha=0.5, label='No Eye Movement')
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.legend()
            st.pyplot(plt)
            plt.clf()

    if st.sidebar.checkbox("Show Boxplots"):
        for col in numerical_columns:
            sns.boxplot(x=pf_1[col])
            plt.title(col)
            st.pyplot(plt)
            plt.clf()

    # Correlation Heatmap
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 10))
        sns.heatmap(pf_1.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)

    # Outlier Detection
    if st.sidebar.checkbox("Show Outliers"):
        z_scores = np.abs(stats.zscore(pf_1.select_dtypes(include=np.number)))
        outliers = np.where(z_scores > 3)
        st.write(f"Outliers detected at: {outliers}")

    # Data Preprocessing and Modeling
    if st.sidebar.checkbox("Run RandomForest Model for Eye Detection"):
        X_eye = pf_1.drop('eyeDetection', axis=1)
        y_eye = pf_1['eyeDetection']
        
        X_train_eye, X_test_eye, y_train_eye, y_test_eye = train_test_split(X_eye, y_eye, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_eye = scaler.fit_transform(X_train_eye)
        X_test_eye = scaler.transform(X_test_eye)
        
        model_eye = RandomForestClassifier(n_estimators=100, random_state=42)
        model_eye.fit(X_train_eye, y_train_eye)
        y_pred_eye = model_eye.predict(X_test_eye)
        
        st.write("### RandomForest Classification Report for Eye Detection")
        st.text(classification_report(y_test_eye, y_pred_eye))
        eye_accuracy = accuracy_score(y_test_eye, y_pred_eye)
        st.write(f"RandomForest Model Accuracy: {eye_accuracy * 100:.2f}%")

    # Stress Level Prediction
    if st.sidebar.checkbox("Run DecisionTree Model for Stress Level Prediction"):
        pf_1['beta-alpha-ratio'] = (pf_1['Beta1'] + pf_1['Beta2']) / (pf_1['Alpha1'] + pf_1['Alpha2'])
        pf_1['stress_level'] = pf_1['beta-alpha-ratio'].apply(lambda x: 2 if x > 1.5 else (1 if x > 1 else 0))
        
        X_stress = pf_1[['Beta1', 'Beta2', 'Alpha1', 'Alpha2']]
        y_stress = pf_1['stress_level']
        X_train_stress, X_test_stress, y_train_stress, y_test_stress = train_test_split(X_stress, y_stress, test_size=0.2, random_state=42)
        
        model_stress = DecisionTreeClassifier(random_state=42)
        model_stress.fit(X_train_stress, y_train_stress)
        y_pred_stress = model_stress.predict(X_test_stress)
        
        st.write("### DecisionTree Classification Report for Stress Level Prediction")
        st.text(classification_report(y_test_stress, y_pred_stress))
        stress_accuracy = accuracy_score(y_test_stress, y_pred_stress)
        st.write(f"DecisionTree Model Accuracy: {stress_accuracy * 100:.2f}%")

    # K-Means Clustering for ADHD Prediction
    if st.sidebar.checkbox("Run K-Means Clustering for ADHD Prediction"):
        X_adhd = pf_1[['Mediation', 'Attention']]
        scaler = StandardScaler()
        X_adhd_scaled = scaler.fit_transform(X_adhd)
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        pf_1['ADHD_Cluster'] = kmeans.fit_predict(X_adhd_scaled)
        
        st.write("### K-Means Clustering Results")
        plt.scatter(pf_1['Mediation'], pf_1['Attention'], c=pf_1['ADHD_Cluster'], cmap='viridis')
        plt.xlabel("Mediation")
        plt.ylabel("Attention")
        plt.title("K-Means Clustering for ADHD Prediction")
        st.pyplot(plt)

        # ADHD Prediction based on clustering
        pf_1['ADHD_Prediction'] = np.where(pf_1['ADHD_Cluster'] == 0, 1, 0)
        st.write("ADHD Prediction (1: ADHD detected, 0: No ADHD)")
        st.write(pf_1[['Mediation', 'Attention', 'ADHD_Prediction']])
