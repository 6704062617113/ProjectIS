import streamlit as st
st.set_page_config(page_title="Project IS", layout="wide")
st.title(" Project IS Dashboard")
st.markdown("## เลือกหน้าที่ต้องการเข้าใช้งาน")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
with col1:
    if st.button(" Machine Learning Model"):
        st.switch_page("pages/Machine_Learning.py")
with col2:
    if st.button("Neural Network Model"):
        st.switch_page("pages/Neural_Network.py")
with col3:
    if st.button("Test ML Model"):
        st.switch_page("pages/Test_ML.py")
with col4:
    if st.button("Test Neural Network"):
        st.switch_page("pages/Test_NN.py")