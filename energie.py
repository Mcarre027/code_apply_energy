import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import gdown
import os  # Ajout de l'import manquant

@st.cache_data
def get_logo():
    file_id = "1ZF4CX_g41jhOjNipe9OhCTB7mnDLn6Ed"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    return BytesIO(response.content)

@st.cache_data
def load_data():
    file_id = "15l7StwyKMtW9dGB-MrnD_hcZtELMCqbz"
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        output = "eco2mix-regional-cons-def.csv"
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)
        
        df = pd.read_csv(output, sep=";", low_memory=False)
        return df
    except Exception as e:
        st.error("Erreur lors du chargement des données")
        st.write(f"Détails de l'erreur : {str(e)}")
        return None

# Affichage du logo
try:
    logo = get_logo()
    st.sidebar.image(logo, width=250)
except Exception:
    st.sidebar.write("Logo non disponible")

# Chargement des données (une seule fois)
df = load_data()
if df is None:
    st.stop()
#création de la variable "production totale 
df['Production_Totale'] = df[['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)',
                                      'Solaire (MW)', 'Hydraulique (MW)', 'Bioénergies (MW)','Pompage (MW)']].sum(axis=1)
#Changement du type de la variable éolien
df['Eolien (MW)'] = pd.to_numeric(df['Eolien (MW)'], errors='coerce')
#préparation pour des heures  graphique
df['Date - Heure'] = pd.to_datetime(df['Date - Heure'])
df.set_index('Date - Heure', inplace=True)

# Titre principal
st.title("Observatoire de la production et consommation électrique en France")

# Votre code pour les visualisations et analyses ici
st.write("### Statistiques et indicateurs")


st.sidebar.title("Sommaire")

pages=["Introduction","Exploration du jeu de données", "Statistiques et indicateurs", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Introduction")
  # Paragraphe d'introduction
  st.write("""
    Dans le cadre de ce projet d'étude énergétique, nous nous intéressons à la problématique critique de la prévention des blackouts sur le réseau électrique français. 
    Cette étude s'inscrit dans un contexte plus large de transition énergétique et de gestion des risques liés à l'équilibre du réseau, 
    particulièrement en période de forte sollicitation ou de conditions météorologiques extrêmes.
    Notre analyse se concentre particulièrement sur l'exploitation des données historiques de consommation et de production fournies par RTE 
    (Réseau de Transport d'Électricité), en corrélation avec les données météorologiques. Cette approche vise à identifier les patterns et 
    facteurs de risque pouvant conduire à des situations de tension sur le réseau électrique.
    L'anticipation des potentielles ruptures d'approvisionnement représente un enjeu majeur 
    pour les gestionnaires de réseau, les fournisseurs d'énergie et les consommateurs. 
    En effet, la capacité à prévoir et prévenir les situations de blackout est cruciale pour maintenir la continuité du service électrique et 
    garantir la sécurité d'approvisionnement de l'ensemble des usagers. Cette problématique est d'autant plus pertinente dans le contexte actuel de transition énergétique, 
    où l'intégration croissante d'énergies renouvelables, plus variables par nature, complexifie la gestion de l'équilibre du réseau.""")
 



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
    # Titre principal
    st.header("Rapport Matthieu")
    
    # Sous-titre
    st.subheader("Analyse des dynamiques de production et consommation électrique en France (2014-2022)")
    
    # Paragraphe d'introduction
    st.write("""
    Lors de nos réunions de cadrage, nous avons identifié les différents axes d'analyse nécessaires pour comprendre 
    en profondeur les dynamiques du réseau électrique français. Pour optimiser notre travail et assurer une couverture 
    exhaustive du sujet, nous avons décidé de répartir les sujets selon les compétences et les centres d'intérêt de 
    chacun. Dans ce contexte, j'ai pris en charge l'analyse approfondie de la relation entre la consommation et la 
    production d'électricité. Cette analyse cruciale vise à comprendre comment le système électrique français s'adapte 
    aux variations de la demande et à identifier les potentielles périodes de tension sur le réseau.
    """)
    
    # Titre de section
    st.header("Évolution Comparative de la Production et de la Consommation Électrique (2012-2023)")
    
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




