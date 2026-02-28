import streamlit as st
st.set_page_config(page_title="Project IS", layout="wide")
st.markdown("<h1 style='text-align: center;'>Project IS</h1>", unsafe_allow_html=True)
st.write("")
st.write("")
row1_left, row1_center, row1_right = st.columns([1,2,1])
row2_left, row2_center, row2_right = st.columns([1,2,1])
with row1_center:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Machine Learning Model", use_container_width=True):
            st.switch_page("pages/Machine_Learning.py")
    with col2:
        if st.button("Neural Network Model", use_container_width=True):
            st.switch_page("pages/Neural_Network.py")
with row2_center:
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Test ML Model", use_container_width=True):
            st.switch_page("pages/Test_ML.py")
    with col4:
        if st.button("Test Neural Network", use_container_width=True):
            st.switch_page("pages/Test_NN.py")