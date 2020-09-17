import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px


data=pd.read_csv('result_data.csv')
data=data.iloc[:,1:]
cfmatx=pd.read_csv('rf_confusion_matrix.csv')
metric_df=pd.read_csv('metric_df.csv')
top5_df=pd.read_csv('top5_df.csv')

st.write("""
Interpretations for the Random Forest Credit Data Predictions: The goal is to obtain a model that
may be used to identify factors that make an applicant at higher risk of default. 
""")

st.subheader('Results')
default_type=st.multiselect('Select a Default Type', list(data.DEFAULT.unique()), default=[0,1])
new_data=data[data.DEFAULT.isin(default_type)]
st.write(new_data.iloc[:,1:])

st.subheader('Chart')
pred=st.selectbox('X', data.columns[0:3], index=2)
deft=st.selectbox('Y', data.columns[0:3], index=1)
fig = px.scatter(new_data, x =pred,y=deft, color='DEFAULT')
st.plotly_chart(fig)

st.subheader('Confusion Matrix')
st.write(cfmatx.iloc[:,1:])

st.subheader('Metrics')
st.write(metric_df.iloc[:,1:])

st.subheader('Top 5 Features by Importance')
st.write(top5_df.iloc[:,1:])

st.write('''The illustration above shows that the predictions for 
positive defaults has greater probablity strengths supporting 
a higher level of confidence in determining the applicant profiles
more likely to default on a credit loan.  Specifically, applicants 
with no checking account or a zero or less checking account balance, and/or
critical account history and applying for a new car loan
or a furniture/equipment loan is more likely to default on a credit
agreement''')
