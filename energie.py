import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("eco2mix-regional-cons-def.csv", sep = ";")

st.title("Observatoire de la production et consommation électrique en France ")
st.sidebar.image("image/logo apply.png",width=250)
st.sidebar.title("Sommaire")

pages=["Introduction","Exploration du jeu de données", "Statistiques et indicateurs", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Introduction")
 



if page == pages[1] : 
    st.write("### Exploration du jeu de données")
    st.markdown("\n\n\n")
    st.write('<u>**Dataset eco2mix régional**</u>',unsafe_allow_html=True)
    st.dataframe(df.head(10))
    st.markdown("\n\n\n")
    st.markdown("\n\n\n")
    st.write('<u>**Dimension du dataset**</u>',unsafe_allow_html=True)
    st.write(df.shape)
    st.markdown("\n\n\n")
    st.markdown("\n\n\n")
    st.write('<u>**Statistiques des variables du dataset**</u>',unsafe_allow_html=True)
    st.dataframe(df.describe())
    st.markdown("\n\n\n")
    st.markdown("\n\n\n")
    st.write('<u>**Affichage des valeurs manquantes**</u>',unsafe_allow_html=True)
    if st.checkbox("Afficher les NA") :
     st.dataframe(df.isna().sum())



if page == pages[2] : 
    st.write("### Statistiques et indicateurs")







if page == pages[3] :
    st.write("### Modélisation")




