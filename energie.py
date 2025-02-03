import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Chargement optimisé du dataset
@st.cache_data
def load_data():
    current_dir = Path.cwd()
    csv_path = current_dir / "eco2mix-regional-cons-def.csv"
    try:
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
st.title("⚡ Prédiction de la consommation électrique en France")

# Sidebar
st.sidebar.title("🗂️ Navigation")
pages = ["📘 Introduction", "🔍 Exploration des données", "📊 Analyse et visualisations", "🤖 Modélisation"]
page = st.sidebar.radio("Aller vers", pages)

# Page Introduction
if page == pages[0]:
    st.write("## 📘 Introduction")
    st.write(
        """
        Cette application présente une analyse des données électriques en France et une démonstration de 
        modèle de machine learning pour prédire la consommation. Vous pourrez explorer les données, 
        examiner des visualisations clés, et comprendre les facteurs influençant la consommation.
        """
    )

# Page Exploration du jeu de données
elif page == pages[1]:
    st.write("## 🔍 Exploration des données")
    if df is not None:
        st.write("### **Aperçu du dataset**")
        st.dataframe(df.head(10))

        st.write("### **Dimension du dataset**")
        st.write(f"Nombre de lignes : {df.shape[0]}, Nombre de colonnes : {df.shape[1]}")

        st.write("### **Statistiques descriptives**")
        st.write(df.describe())

        st.write("### **Colonnes avec valeurs manquantes**")
        missing_values = df.isna().sum()
        cols_with_na = missing_values[missing_values > 0]
        if not cols_with_na.empty:
            st.write(cols_with_na)
        else:
            st.write("Aucune valeur manquante détectée.")
    else:
        st.error("Le dataset n'a pas pu être chargé.")

# Page Analyse et visualisations
elif page == pages[2]:
    st.write("## 📊 Analyse et visualisations")

    if df is not None:
        # Convertir les colonnes en numérique
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
        st.write("### 📅 Production vs Consommation Totale par Année")
        comparaison = df[["Date - Heure", "Production_Totale", "Consommation (MW)"]].set_index("Date - Heure")
        comparaison_annuelle = comparaison.resample("Y").mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        comparaison_annuelle.plot(kind="bar", ax=ax, color=["blue", "orange"])
        ax.set_xlabel("Année")
        ax.set_ylabel("Moyenne annuelle (MW)")
        st.pyplot(fig)

        # Matrice de corrélation
        st.write("### 📈 Matrice de corrélation")
        colonnes_analyse = ["Consommation (MW)", "Production_Totale"] + colonnes_production
        correlation_matrix = df[colonnes_analyse].corr()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

# Page Modélisation
elif page == pages[3]:
    st.write("## 🤖 Modélisation")

    st.write(
        """
        Cette section présente une démonstration de modèle de machine learning pour prédire la consommation électrique
        à partir des données de production. Le modèle sera entraîné avec les données disponibles, et vous pourrez
        visualiser les prédictions sur les séries temporelles.
        """
    )

    # Préparation des données pour modélisation (exemple minimal)
    if df is not None:
        st.write("### 🛠️ Préparation des données pour la modélisation")
        st.write("Les colonnes de production sont utilisées comme variables explicatives.")

        # Afficher un aperçu des données utilisées
        st.dataframe(df[["Date - Heure", "Consommation (MW)", "Production_Totale"]].head(10))

    else:
        st.error("Le dataset n'est pas disponible.")

