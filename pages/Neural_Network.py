import streamlit as st
st.title("Neural Network Model")
st.write("#### **Data Prepartion**")
st.write("**Data Form:** https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease?utm_source=chatgpt.com")
st.write("""**Feature:**        
- **Sex:** Sex (0 = female, 1 = male) 
- **Cp:** 0(Typical Angina) 1(Atypical Angina) 2(Non-anginal Pain) 3(Asymptomatic)
- **trestbps:** Resting Blood Pressure (in mm Hg)
- **chol:** Serum Cholesterol (in mg/dl)
- **fbs:** Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg:** Resting Electrocardiographic Results 0(Normal) 1(ST-T wave abnormality) 2(Left ventricular hypertrophy) 
- **thalach:** Maximum Heart Rate Achieved  
- **exang:** Exercise Induced Angina (1 = yes, 0 = no)
- **oldpeak:** ST depression induced by exercise relative to rest
- **slope:** Slope of the peak exercise ST segment 0(Upsloping) 1(Flat) 2(Downsloping)
- **ca:** Number of major vessels (0-3) colored by fluoroscopy
- **thal:** 0(Normal)  1(Fixed defect) 2(Reversible defect)
- **target (Target Variable):** Presence of heart disease   0 = No heart disease ,1-4 = Presence of heart disease""")
st.write("""**Data Cleaning:**
         
Missing values represented as “?” were replaced with the median of each feature. The target variable was converted into binary classes (0 = No Heart Disease, 1 = Heart Disease). All features were standardized using StandardScaler before training the neural network.""")
st.write("#### **Algorithms**")
st.write("**Artificial Neural Networks**")
st.write("""ANNs work by learning patterns in data through a process called training. During training, the network adjusts itself to improve its accuracy by comparing its predictions with the actual results.\n
**Key Components of an ANN**
Input Layer: This is where the network receives information. For example, in an image recognition task, the input could be an image.
Hidden Layers: These layers process the data received from the input layer. The more hidden layers there are, the more complex patterns the network can learn and understand. Each hidden layer transforms the data into more abstract information.
Output Layer: This is where the final decision or prediction is made. For example, after processing an image, the output layer might decide whether it’s a cat or a dog.""")
st.write("#### **Model Development Process**")
st.write("""The model development process consists of the following steps:
1.	Load the dataset and assign column names.
2.	Clean the data by replacing missing values with the median and converting the target into binary form (0 = no disease, 1 = disease).
3.	Split the dataset into Training, Validation, and Test sets.
4.	Apply feature scaling using StandardScaler to normalize input features.
5.	Build a Multilayer Perceptron (MLP) with multiple dense layers.
6.	Train the model using Binary Cross-Entropy loss, Adam optimizer, and Early Stopping.
7.	Evaluate model performance using Accuracy and Recall.
8.	Select the optimal classification threshold.
9.	Save the trained model and scaler for deployment in the web application.""")
st.write("#### **References**")
st.write("""**Data:** https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease?utm_source=chatgpt.com\n
**Information about ANN:** https://www.geeksforgeeks.org/deep-learning/artificial-neural-networks-and-its-applications/""")