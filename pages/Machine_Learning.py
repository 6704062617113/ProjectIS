import streamlit as st
st.title(" Machine Learning Model")
st.write("#### **Data Prepartion**")
st.write("**Data Form:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
st.write("""**Feature:**        
- **Pregnancies:** Number of times pregnant  
- **Glucose:** Plasma glucose concentration 2 hours in an oral glucose tolerance test  
- **BloodPressure:** Diastolic blood pressure (mm Hg)  
- **SkinThickness:** Triceps skin fold thickness (mm)  
- **Insulin:** 2-Hour serum insulin (mu U/ml)  
- **BMI:** Body mass index (weight in kg/(height in m)^2)  
- **DiabetesPedigreeFunction:** A function that represents the likelihood of diabetes based on family history. Higher values indicate a stronger genetic predisposition.  
- **Age:** Age in years  
- **Outcome (Target Variable):** 0(No diabetes) 1(Diabetes)  """)
st.write("""**Data Cleaning:**
         
Some features (Glucose, BloodPressure, SkinThickness, Insulin, and BMI) contain zero values that are medically unrealistic and indicate missing data.
The following preprocessing steps were applied:
1.	Replace zero values with NaN (missing values).
2.	Impute missing values using the median of each feature.\n
Median imputation was selected because medical datasets may contain outliers.""")
st.write("""**Train-test Split:**  80% Training Set and 20% Testing Set""")
st.write("""**Standard Scaler:**  Use StandardScaler to adjust attribute values ​​to standard format (Mean = 0, Standard Deviation = 1).""")
st.write("#### **Algorithms**")
st.write("The model developed uses Ensemble Learning, which combines several algorithms, including:")
st.write("""**1.K-Nearest Neighbors (KNN):** \n
K-Nearest Neighbors (KNN) is a supervised machine learning algorithm generally used for classification but can also be used for regression tasks. It works by finding the "k" closest data points (neighbors) to a given input and makes a predictions based on the majority class (for classification) or the average value (for regression). Since KNN makes no assumptions about the underlying data distribution it makes it a non-parametric and instance-based learning method.\n
**2.Support Vector Machine (SVM):** \n
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It tries to find the best boundary known as hyperplane that separates different classes in the data. It is useful when you want to do binary classification like spam vs. not spam or cat vs. dog.\n
The main goal of SVM is to maximize the margin between the two classes. The larger the margin the better the model performs on new and unseen data.\n
**3.Random Forest:**\n
Random Forest is a machine learning algorithm that uses many decision trees to make better predictions. Each tree looks at different random parts of the data and their results are combined by voting for classification or averaging for regression which makes it as ensemble learning technique. This helps in improving accuracy and reducing errors.""")
st.write("#### **Model Development Process**")
st.write("""
1.	Load dataset from CSV file
2.	Perform data cleaning and missing value imputation
3.	Separate features (X) and target variable (y)
4.	Split data into training and testing sets
5.	Apply feature scaling
6.	Train KNN, SVM, and Random Forest models
7.	Combine models using Voting Classifier
8.	Train the ensemble model
9.	Evaluate model performance using accuracy
10.	Save the trained model and scaler for deployment""")
st.write("#### **References**")
st.write("""**Data:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n
**Information about KNN:** https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/\n
**Information about SVM:** https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/\n
**Information about Random Forest:** https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/""")