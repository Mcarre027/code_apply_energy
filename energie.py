import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import gdown




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
    
       


if page == pages[2]:
    st.write("### Statistiques et indicateurs")
    
    # Création du DataFrame de comparaison
    comparaison = pd.DataFrame({
        'Production Totale': df['Production_Totale'],
        'Consommation': df['Consommation (MW)']
    })
    
    # Calcul des moyennes annuelles
    comparaison_annuelle = comparaison.resample('Y').mean()
    
    # Définition des couleurs
    color_production = 'blue'
    color_consommation = 'orange'
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(16, 8))
    
    comparaison_annuelle.plot(
        kind='bar',
        color=[color_production, color_consommation],
        width=0.8,
        ax=ax
    )
    
    # Personnalisation du graphique
    ax.set_title('Comparaison Production vs Consommation Totale par Année',
                 fontsize=16,
                 pad=20)
    ax.set_xlabel('Année', fontsize=12)
    ax.set_ylabel('Moyenne annuelle (MW)', fontsize=12)
    
    ax.set_xticklabels([label.get_text()[:4] for label in ax.get_xticklabels()],
                       rotation=45, ha='right')
    
    ax.legend(bbox_to_anchor=(1.05, 1),
              loc='upper left',
              fontsize=12)
    
    ax.grid(True, alpha=0.2, axis='y')
    
    # Ajustement de la mise en page
    fig.tight_layout()
    
    # Affichage du graphique dans Streamlit
    st.pyplot(fig)
    
    # Affichage des statistiques
    st.write("\nStatistiques annuelles :")
    st.dataframe(comparaison_annuelle.describe())







if page == pages[3] :
    st.write("### Modélisation")




