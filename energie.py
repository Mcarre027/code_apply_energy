import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import gdown


df2 = pd.read_excel("Rapport exploration des données sujet energie.xlsx")

@st.cache_data
def get_logo():
    file_id = "1ZF4CX_g41jhOjNipe9OhCTB7mnDLn6Ed"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    return BytesIO(response.content)

# Chargement des données
@st.cache_data
def load_data():
    file_id = "1QyqA7mJ68MiKQ7-7p-LTmrRNmRFAl5IJ"
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        output = "eco2mix-regional-cons-def.csv"
        gdown.download(url, output, quiet=False)
        return pd.read_csv(output, sep=";")
    except Exception as e:
        st.error("Erreur lors du chargement des données")
        raise e

# Affichage du logo
try:
    logo = get_logo()
    st.sidebar.image(logo, width=250)
except Exception:
    st.sidebar.write("Logo non disponible")

# Chargement et utilisation des données
df = load_data()
st.title("Observatoire de la production et consommation électrique en France ")


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
    st.markdown("\n\n\n")
    st.markdown("\n\n\n")
    st.write('<u>**Affichage des valeurs manquantes**</u>',unsafe_allow_html=True)
    if st.checkbox("Rapport exploration des données sujet energie") :
     st.dataframe(df2.head(32))
       


if page == pages[2] : 
    st.write("### Statistiques et indicateurs")







if page == pages[3] :
    st.write("### Modélisation")




