import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path

# Chargement optimisé du dataset
@st.cache_data
def load_data():
    current_dir = Path.cwd()
    csv_path = current_dir / "eco2mix-regional-cons-def.csv"
    try:
        # Charger uniquement les colonnes nécessaires pour l'analyse
        cols = [
            "Date - Heure", "Région", "Consommation (MW)", "Thermique (MW)", "Nucléaire (MW)", 
            "Eolien (MW)", "Solaire (MW)", "Hydraulique (MW)", "Bioénergies (MW)", "Pompage (MW)"
        ]
        df = pd.read_csv(csv_path, sep=';', usecols=cols, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
        return None

# Chargement du dataset
df = load_data()

# Interface Streamlit
st.title("Observatoire de la production et consommation électrique en France")

# Sidebar
st.sidebar.title("Sommaire")
pages = ["📘 Introduction", "🔍 Exploration du jeu de données", "📊 Statistiques et indicateurs", "🤖 Modélisation"]
page = st.sidebar.radio("Aller vers", pages)

# Page Introduction
if page == pages[0]:
    st.write("### 📘 Introduction")
    st.write(
        """
        Ce tableau de bord interactif permet d'explorer et d'analyser les données sur la production et la consommation
        d'électricité en France. Vous pourrez y trouver des statistiques descriptives, des visualisations et des 
        modèles prédictifs pour mieux comprendre les tendances énergétiques.
        """
    )

# Page Exploration du jeu de données
elif page == pages[1]: 
    st.write("### 🔍 Exploration du jeu de données")

    if df is not None:
        st.write("#### **Aperçu du dataset**")
        st.dataframe(df.head(10))

        st.write("#### **Dimension du dataset**")
        st.write(df.shape)

        if st.checkbox("Afficher les statistiques descriptives"):
            st.write(df.describe())

        if st.checkbox("Afficher les colonnes avec des valeurs manquantes"):
            missing_values = df.isna().sum()
            cols_with_na = missing_values[missing_values > 0]
            st.write(cols_with_na)
    else:
        st.error("Le dataset n'a pas pu être chargé.")

# Page Statistiques et indicateurs
elif page == pages[2]:
    st.write("### 📊 Statistiques et indicateurs")

    if df is not None:
        # Convertir les colonnes en numérique pour les calculs
        colonnes_production = ["Thermique (MW)", "Nucléaire (MW)", "Eolien (MW)", "Solaire (MW)", 
                               "Hydraulique (MW)", "Bioénergies (MW)"]

        for col in colonnes_production:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculer la production totale
        df["Production_Totale"] = df[colonnes_production].sum(axis=1)

        # Conversion de la colonne date
        df["Date - Heure"] = pd.to_datetime(df["Date - Heure"], errors="coerce")
        df["Année"] = df["Date - Heure"].dt.year

        # Comparaison production vs consommation
        if st.checkbox("Afficher le graphique de comparaison annuelle"):
            comparaison = df[["Date - Heure", "Production_Totale", "Consommation (MW)"]].set_index("Date - Heure")
            comparaison_annuelle = comparaison.resample("Y").mean()

            st.write("#### Comparaison Production vs Consommation Totale par Année")
            fig, ax = plt.subplots(figsize=(12, 6))
            comparaison_annuelle.plot(kind="bar", ax=ax, color=["blue", "orange"])
            ax.set_title("Comparaison Production vs Consommation Totale par Année")
            ax.set_xlabel("Année")
            ax.set_ylabel("Moyenne annuelle (MW)")
            st.pyplot(fig)

        # Matrice de corrélation
        if st.checkbox("Afficher la matrice de corrélation"):
            colonnes_analyse = ["Consommation (MW)", "Production_Totale"] + colonnes_production
            correlation_matrix = df[colonnes_analyse].corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
            ax.set_title("Matrice de corrélation entre consommation et production")
            st.pyplot(fig)


# Page Modélisation
elif page == pages[3]:
    st.write("### 🤖 Modélisation")
