import streamlit as st
import pickle
import numpy as np
import os

# ---------- Load Model (Cloud Fix) ----------
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path,'rb'))

# ---------- Placeholder Style ----------
st.markdown("""
    <style>
    input[type=number]::-webkit-input-placeholder {
        color: #aaa;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Placement Package Predictor")

cgpa = st.number_input("Enter CGPA",
                       min_value=0.0,
                       max_value=10.0,
                       value=None,
                       placeholder="0.00",
                       step=0.1)

projects = st.number_input("Projects Completed",
                           min_value=0,
                           max_value=10,
                           value=None,
                           placeholder="0")

internships = st.number_input("Internships Done",
                              min_value=0,
                              max_value=5,
                              value=None,
                              placeholder="0")

coding = st.slider("Coding Skill",1,10)
communication = st.slider("Communication Skill",1,10)

if st.button("Predict Package"):
    if cgpa is not None and projects is not None and internships is not None:
        input = np.array([[cgpa,projects,internships,coding,communication]])
        prediction = model.predict(input)
        st.success(f"Expected Package: {prediction[0]:.2f} LPA")
    else:
        st.warning("Please enter all values")

