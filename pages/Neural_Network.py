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
st.write("**Multi-Layer Perceptron**")
st.write("""Multi-Layer Perceptron (MLP) consists of fully connected dense layers that transform input data from one dimension to another. It is called multi-layer because it contains an input layer, one or more hidden layers and an output layer. The purpose of an MLP is to model complex relationships between inputs and outputs.\n
**Components of Multi-Layer Perceptron (MLP)**\n
Input Layer: Each neuron or node in this layer corresponds to an input feature. For instance, if you have three input features the input layer will have three neurons.\n
Hidden Layers: MLP can have any number of hidden layers with each layer containing any number of nodes. These layers process the information received from the input layer.\n
Output Layer: The output layer generates the final prediction or result. If there are multiple outputs, the output layer will have a corresponding number of neurons.\n
every node in one layer connects to every node in the next layer. As the data moves through the network each layer transforms it until the final output is generated in the output layer.\n
**Working of Multi-Layer Perceptron**\n
1. Forward Propagation : the data flows from the input layer to the output layer, passing through any hidden layers. Each neuron in the hidden layers processes the input\n
2. Loss Function : Once the network generates an output the next step is to calculate the loss using a loss function. In supervised learning this compares the predicted output to the actual label.\n
3. Backpropagation : The goal of training an MLP is to minimize the loss function by adjusting the network's weights and biases.""")
st.write("#### **Model Development Process**")
st.write("""The model development process consists of the following steps:
1.	Load and preprocess the dataset by handling missing values with the median and converting the target into binary form.
2.	Split the data into training, validation, and test sets.
3.	Apply feature scaling using StandardScaler.
4.	Build a Multilayer Perceptron (MLP) model.
5.	Train the model using Binary Cross-Entropy loss, Adam optimizer, and Early Stopping.
6.	Evaluate the model using Accuracy and Recall.
7.	Select the optimal classification threshold.	
8.	Save the trained model and scaler for deployment.
""")
st.write("#### **References**")
st.write("""**Data:** https://uci-ics-mlr-prod.aws.uci.edu/dataset/45/heart%2Bdisease?utm_source=chatgpt.com\n
**Information about MLP:** https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/""")