import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

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
        if not csv_path.exists():
            st.error(f"Fichier CSV introuvable : {csv_path}")
            return None
        df = pd.read_csv(csv_path, sep=';', usecols=cols, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
        return None

# Chargement des données
df = load_data()

# Titre principal
st.title("⚡ Prédiction de la consommation électrique en France")

# Navigation
with st.sidebar:
    st.title("🗂️ Navigation")
pages = ["📘 Introduction", "🔍 Exploration des données", "📊 Analyse", "🤖 Modélisation"]
page = st.sidebar.radio("Aller vers", pages)

# Page Introduction
if page == pages[0]:
    st.write("## 📘 Introduction")
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel
    st.write("""
        Cette application présente une analyse des données électriques en France et une démonstration 
        de modèle de machine learning pour prédire la consommation.
    """)
    st.write("")  # Espace visuel
    # 🔹 Chargement et affichage de la vidéo locale
    video_path = "20250309_0113_Blend Video_blend_01jnw3zxa8ebhtsdjmkk4m6j3r.mp4"  
    if os.path.exists(video_path):
        st.video(video_path, start_time=0)
    
        
    

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
                               "Hydraulique (MW)", "Bioénergies (MW)", "Pompage (MW)"]
        for col in colonnes_production:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df["Production_Totale"] = df[colonnes_production].sum(axis=1)
        df["Date - Heure"] = pd.to_datetime(df["Date - Heure"], errors="coerce")
        st.write("")  # Espace visuel
        # Comparaison production vs consommation
        st.write("## Production vs Consommation")
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
        st.write("## Corrélation")
        colonnes_analyse = ["Consommation (MW)", "Production_Totale"] + colonnes_production
        correlation_matrix = df[colonnes_analyse].corr()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
        st.write("")  # Espace visuel
        st.write("")  # Espace visuel
        st.write("### Justification des variables utilisées pour prédire la consommation")
        st.write("")  # Espace visuel
        st.write("""
         Sélection des Variables pour la Modélisation

La matrice de corrélation montre que les différentes sources de production d'énergie entretiennent des relations variées avec la consommation.  
Nous avons sélectionné les variables suivantes pour la prédiction de la consommation énergétique :  
**Thermique**, **Nucléaire**, **Éolien**, **Solaire**, **Hydraulique**, **Bioénergies**, et **Pompage**.

#### 1. Variables fortement corrélées à la consommation
Certaines sources de production montrent une corrélation significative avec la consommation :  
- Les **bioénergies** (corrélation : **0.59**),  
- L'**hydraulique** (corrélation : **0.44**),  
- Le **thermique** (corrélation : **0.33**).  

Ces variables contribuent directement à expliquer les variations de la consommation.

#### 2. Production nucléaire
La **production nucléaire** est stable mais reste un facteur clé avec une corrélation de **0.21**.

#### 3. Énergies renouvelables et variations saisonnières
Bien que les énergies renouvelables présentent des corrélations plus faibles, elles capturent efficacement les variations saisonnières :  
- **Éolien** (corrélation : **0.059**),  
- **Solaire** (corrélation : **0.04**).

#### 4. Effet du pompage et stockage d'énergie
La variable **Pompage** a une corrélation négative (**-0.19**), ce qui reflète son rôle dans le stockage d'énergie, entraînant un effet inverse sur la consommation.

#### Conclusion
Ces variables permettent de représenter les dynamiques complexes entre production et consommation énergétique, tout en prenant en compte les variations saisonnières, les fluctuations de production et les effets liés au stockage d'énergie. Ce choix améliore la précision des prédictions du modèle.
""")


                 
    else:
        st.error("Les données n'ont pas pu être chargées.")

elif page == pages[3]:
    st.subheader("🤖 Modélisation")
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel
    
    if df is not None:
        predictions_file = "predictions_2019.csv"
        
        # 📌 Préparer les données
        X_production = df[["Thermique (MW)", "Nucléaire (MW)", "Eolien (MW)", "Solaire (MW)", 
                           "Hydraulique (MW)", "Bioénergies (MW)", "Pompage (MW)"]]
        X_production = X_production.replace(['ND', '-'], 0).fillna(0)

        # 📌 Chargement des prédictions 2019
        if os.path.exists(predictions_file):
            df_filtered = pd.read_csv(predictions_file, parse_dates=["Date - Heure"])
        else:
            st.error("Le fichier de prédictions 2019 n'a pas été trouvé.")
            df_filtered = pd.DataFrame()

        # 📌 Affichage des résultats 2019
        if not df_filtered.empty:
            st.subheader("📊 Comparaison entre consommation réelle et prédite (2019)")
            df_filtered["Mois"] = df_filtered["Date - Heure"].dt.to_period("M")
            df_grouped = df_filtered.groupby("Mois")[["Consommation (MW)", "Consommation Prédite"]].mean().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_grouped["Mois"].astype(str), df_grouped["Consommation (MW)"], width=0.4, label="Consommation réelle", color="blue", align='center')
            ax.bar(df_grouped["Mois"].astype(str), df_grouped["Consommation Prédite"], width=0.4, label="Consommation prédite", color="orange", align='edge')
            ax.set_xlabel("Mois")
            ax.set_ylabel("Consommation moyenne (MW)")
            ax.set_title("Comparaison entre consommation réelle et prédite par mois (2019)")
            ax.legend(loc="upper right")
            plt.xticks(rotation=45)
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

            # 📌 Affichage des métriques
            df_eval = df_filtered[["Consommation (MW)", "Consommation Prédite"]].dropna()
            mse = mean_squared_error(df_eval["Consommation (MW)"], df_eval["Consommation Prédite"])
            r2 = r2_score(df_eval["Consommation (MW)"], df_eval["Consommation Prédite"])
            st.write(f"📉 **Erreur quadratique moyenne (MSE) :** {mse:.2f}")
            st.write(f"📈 **Score R² :** {r2:.2f}")

    # 📌 CHARGEMENT DES PRÉDICTIONS XGBoost POUR 2030
            try:
                df_pred = pd.read_csv("predictions_2030.csv", encoding="utf-8", sep=",")  # Charger le fichier

                # ✅ Renommer correctement les colonnes
                df_pred.rename(columns={'Date': 'date', 'Prévision Consommation (MW)': 'xgboost'}, inplace=True)

                # ✅ Convertir la colonne date en datetime
                df_pred['date'] = pd.to_datetime(df_pred['date'])

                # Vérification des colonnes après renommage
                print("🔍 Colonnes après renommage :", df_pred.columns.tolist())

            except FileNotFoundError:
                st.error("❌ Fichier `predictions_2030.csv` introuvable.")
                df_pred = None


    # 🔮 Gros titre pour la section prédictions mensuelles jusqu'en 2030
    st.markdown("""
        <h1 style="text-align: center; color: #1E3A8A; font-size: 36px;">
            🔮 Projection Énergétique : Prédictions Mensuelles Jusqu’en 2030 ⚡
        </h1>
    """, unsafe_allow_html=True)

    if df_pred is not None:
        # 📌 Sélection d'une année (2024-2030)
        year_selected = st.selectbox("📅 Sélectionnez une année :", list(range(2024, 2031)))

        # 📌 Sélection d'un mois
        month_selected = st.selectbox("📆 Sélectionnez un mois :", list(range(1, 13)))

        # 📌 Filtrer les prédictions pour le mois et l'année sélectionnés
        filtered_df = df_pred[
            (df_pred['date'].dt.year == year_selected) & 
            (df_pred['date'].dt.month == month_selected)
        ]

        # 📌 Affichage de la prévision
        if not filtered_df.empty:
            pred_value = filtered_df['xgboost'].values[0]
            st.metric(label=f"📊 Prédiction XGBoost pour {month_selected}/{year_selected}", value=f"{pred_value:.2f} MW")
        else:
            st.warning("⚠️ Aucune donnée disponible pour ce mois/année.")

        # 📌 Graphique des prédictions mensuelles
        fig = px.line(
            df_pred,
            x='date',
            y='xgboost',
            labels={'date': "Date", 'xgboost': "Consommation (MW)"},
            title="📊 Évolution des Prédictions Mensuelles de Consommation (MW) - XGBoost",
            color_discrete_sequence=["#D97706"],  # Orange
        )
        st.plotly_chart(fig)
       
