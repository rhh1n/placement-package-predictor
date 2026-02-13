import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------- Train Model in Cloud ----------
data = {
    'cgpa':[6,7,8,9,7.5,8.5,6.5,9.2,8.8,7.2],
    'projects':[1,2,3,4,2,3,1,5,4,2],
    'internships':[0,1,2,2,1,2,0,3,2,1],
    'coding':[4,6,8,9,7,8,5,9,9,6],
    'communication':[5,6,7,8,6,7,5,8,9,6],
    'package':[3,5,7,12,6,9,4,15,13,5]
}

df = pd.DataFrame(data)

X = df.drop('package',axis=1)
y = df['package']

model = LinearRegression()
model.fit(X,y)

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
