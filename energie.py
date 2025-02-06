import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import parallel_backend

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

# Charger le modèle pré-entraîné
@st.cache_resource
def load_model():
    try:
        return joblib.load("random_forest_model.pkl")
    except FileNotFoundError:
        st.error("Le modèle n'a pas été trouvé.")
        return None

# Charger les données et le modèle
df = load_data()
model = load_model()


# Titre principal
st.title("⚡ Prédiction de la consommation électrique en France")

# **Ajout de la navigation dans la barre latérale**
with st.sidebar:
    st.title("🗂️ Navigation")

# Navigation
pages = ["📘 Introduction", "🔍 Exploration des données", "📊 Analyse", "🤖 Modélisation"]
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
    st.write("## 🔍 Exploration")
    st.write("")  # Espace visuel
    if df is not None:
        st.write("### Aperçu du dataset")
        st.dataframe(df.sample(10))

        st.write("")  # Espace visuel

        st.write("### Dimensions")
        st.write(f"Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")

        st.write("")  # Espace visuel

        st.write("### Statistiques descriptives")
        st.write(df.describe())

        st.write("")  # Espace visuel

        st.write("### Valeurs manquantes")
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
    st.write("## 📊 Analyse")
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel

    if df is not None:
        # Convertir les colonnes en numérique
        colonnes_production = ["Thermique (MW)", "Nucléaire (MW)", "Eolien (MW)", "Solaire (MW)", 
                               "Hydraulique (MW)", "Bioénergies (MW)","Pompage (MW)"]
        for col in colonnes_production:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculer la production totale
        df["Production_Totale"] = df[colonnes_production].sum(axis=1)

        # Conversion de la colonne date
        df["Date - Heure"] = pd.to_datetime(df["Date - Heure"], errors="coerce")
        df["Année"] = df["Date - Heure"].dt.year
        df["Mois"] = df["Date - Heure"].dt.month

        st.write("")  # Espace visuel

        # Comparaison production vs consommation
        st.write("### Production vs Consommation")
        comparaison = df[["Date - Heure", "Production_Totale", "Consommation (MW)"]].set_index("Date - Heure")
        comparaison_annuelle = comparaison.resample("Y").mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        comparaison_annuelle.plot(kind="bar", ax=ax, color=["blue", "orange"])
        ax.set_xlabel("Année")
        ax.set_ylabel("Moyenne annuelle (MW)")
        st.pyplot(fig)

        st.write("")  # Espace visuel
        st.write("")  # Espace visuel
        st.write("")  # Espace visuel

        # Matrice de corrélation
        st.write("### Corrélation")
        colonnes_analyse = ["Consommation (MW)", "Production_Totale"] + colonnes_production
        correlation_matrix = df[colonnes_analyse].corr()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
        st.write("")  # Espace visuel
        st.write("")  # Espace visuel

        st.write("## Justification des variables utilisées pour prédire la consommation")
        st.write("")  # Espace visuel

        st.write("""
                    La matrice de corrélation montre que les différentes sources de production d'énergie ont des relations variées avec la consommation. 
                    Nous avons choisi les variables suivantes pour la prédiction : **Thermique**, **Nucléaire**, **Éolien**, **Solaire**, **Hydraulique**, 
                    **Bioénergies**, et **Pompage**. 

                    - Les sources telles que les **bioénergies** (corrélation : 0.59), l'**hydraulique** (0.44) et le **thermique** (0.33) 
                    sont fortement corrélées à la consommation.
                    - La **production nucléaire** est stable mais reste un facteur clé (corrélation : 0.21).
                         Bien que les énergies renouvelables comme l'**éolien** (0.059) et le **solaire** (0.04) aient des corrélations plus faibles, elles permettent de capturer les variations saisonnières.
                    - Enfin, la variable **Pompage** (corrélation : -0.19) est utile pour modéliser les effets inverses liés au stockage d'énergie.

                    Ces variables permettent ainsi de mieux représenter les dynamiques entre production et consommation énergétique.
                """)


# Page Modélisation
elif page == pages[3]:
    st.write("## 🤖 Modélisation")
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel
    if model is not None:
        st.write(
            """
            Cette section présente une démonstration du modèle de machine learning entraîné pour prédire 
            la consommation électrique de l'année 2019 à partir des données de production.
            """
        )
        st.write("")  # Espace visuel
        st.write("")  # Espace visuel
        st.write("### Visualisation des prédictions")
        st.write("")  # Espace visuel
        st.write("")  # Espace visuel
        st.write("")  # Espace visuel

        # Chemin du fichier CSV pour la sauvegarde
        predictions_file = "predictions_2019.csv"
        def load_or_predict(df, model):
            if os.path.exists(predictions_file):
               return joblib.load(predictions_file) 

            

        # Préparer les données pour la prédiction
        X_production = df[["Thermique (MW)", "Nucléaire (MW)", "Eolien (MW)", "Solaire (MW)", 
                               "Hydraulique (MW)", "Bioénergies (MW)","Pompage (MW)"]]

        X_production = X_production.replace('ND', 0)    
        X_production = X_production.replace('-', 0) 
        X_production = X_production.fillna(0)
        df["Date - Heure"] = pd.to_datetime(df["Date - Heure"], errors='coerce')

                  

        # Faire les prédictions
        y_pred = model.predict(X_production)

        # Ajouter les prédictions au DataFrame
        df["Consommation Prédite"] = y_pred

        # Filtrer les données pour une visualisation plus claire (par exemple, une année spécifique)
        df_filtered = df[df["Date - Heure"].dt.year == 2019]
        df_filtered["Mois"] = df_filtered["Date - Heure"].dt.to_period("M")
        df_grouped = df_filtered.groupby("Mois")[["Consommation (MW)", "Consommation Prédite"]].mean().reset_index()

        

        # Visualisation des consommations réelle et prédite
        st.write("### Comparaison entre consommation réelle et prédite (année 2019)")

        # Création du graphique avec des dimensions adaptées
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df_grouped["Mois"].astype(str), df_grouped["Consommation (MW)"], width=0.4, label="Consommation réelle", color="blue", align='center')
        ax.bar(df_grouped["Mois"].astype(str), df_grouped["Consommation Prédite"], width=0.4, label="Consommation prédite", color="orange", align='edge')
        ax.set_xlabel("Mois", fontsize=12)
        ax.set_ylabel("Consommation moyenne (MW)", fontsize=12)
        ax.set_title("Comparaison entre consommation réelle et prédite par mois (2019)", fontsize=14)
        ax.legend(loc="upper right", fontsize=12)
        plt.xticks(rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)


        st.write("")  # Espace visuel

        # Affichage des métriques d'évaluation du modèle
        st.write("### Évaluation du modèle")
        df_eval = df_filtered[["Consommation (MW)", "Consommation Prédite"]].dropna()
        mse = mean_squared_error(df_eval["Consommation (MW)"], df_eval["Consommation Prédite"])
        r2 = r2_score(df_eval["Consommation (MW)"], df_eval["Consommation Prédite"])
        st.write(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
        st.write(f"Score R² : {r2:.2f}")
