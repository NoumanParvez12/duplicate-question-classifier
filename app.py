import streamlit as st
import helper

from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model('model.json')

st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1, q2)

    print("QUESTION 1:", q1)
    print("QUESTION 2:", q2)
    print("FEATURE SHAPE:", query.shape)
    print("PREDICTION:", model.predict(query))
    print("PROBABILITY:", model.predict_proba(query))

    result = model.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')
