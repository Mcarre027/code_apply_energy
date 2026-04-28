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
pages = ["📘 Introduction", "🔍 Exploration des données", "📊 Analyse", "🤖 Modélisation", "🔚 Conclusion générale"]
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

    st.write("")  # Espace visuel
    st.write("")  # Espace visuel

    st.write("""
    ---

    🎯 **Objectif principal :** Prévoir la consommation et les risques de blackout sur le réseau électrique français en croisant les données de consommation, de production et les conditions météorologiques.

    🔎 **Contexte énergétique :** Le projet s’inscrit dans la transition énergétique liée à l’équilibre du réseau, notamment face aux pics de demande, aux conditions climatiques extrêmes ou aux problèmes de production.

    📊 **Données exploitées :** Utilisation des historiques fournis par RTE avec les variables : région, consommation, production par filière.

    🎯 **Enjeu stratégique :** Éviter les coupures d’électricité, surtout avec l’essor des énergies renouvelables intermittentes et les problématiques des centrales nucléaires.

    👥 **Compétences mobilisées :** Le projet s’appuie sur une équipe de 4 personnes avec des rôles et responsabilités sur certaines visualisations ou modèles.

    📦 **Livrables :** Exploration / Data Visualisation / Corrélation variable / Modèle prédictif ML
    """)
        
    

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
        comparaison_annuelle = comparaison.resample("YE").mean()

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

La matrice de corrélation montre que les différentes sources de production d'énergie entretiennent des relations variées avec la consommation.  
Nous avons sélectionné les variables suivantes pour la prédiction de la consommation énergétique :  
**Thermique**, **Nucléaire**, **Éolien**, **Solaire**, **Hydraulique**, **Bioénergies**, et **Pompage**.

#### Variables corrélées à la consommation
Certaines sources de production montrent une corrélation significative avec la consommation :  
- Les **bioénergies** (corrélation : **0.59**),  
- L'**hydraulique** (corrélation : **0.44**),  
- Le **thermique** (corrélation : **0.33**).  

Ces variables contribuent directement à expliquer les variations de la consommation.

#### Production nucléaire
La **production nucléaire** est stable mais reste un facteur clé avec une corrélation de **0.21**.

####  Énergies renouvelables et variations saisonnières
Bien que les énergies renouvelables présentent des corrélations plus faibles, elles capturent efficacement les variations saisonnières :  
- **Éolien** (corrélation : **0.059**),  
- **Solaire** (corrélation : **0.04**).

####  Effet du pompage et stockage d'énergie
La variable **Pompage** a une corrélation négative (**-0.19**), ce qui reflète son rôle dans le stockage d'énergie, entraînant un effet inverse sur la consommation.

#### Conclusion
Ces variables permettent de représenter les dynamiques complexes entre production et consommation énergétique, tout en prenant en compte les variations saisonnières, les fluctuations de production et les effets liés au stockage d'énergie. Ce choix améliore la précision des prédictions du modèle.
""")


                 
    else:
        st.error("Les données n'ont pas pu être chargées.")

elif page == pages[3]:
    st.write("## 🤖 Modélisation")
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel
    
    st.markdown("""
#### Choix des modèles :

Dans le cadre de ce projet, nous avons testé et utilisé plusieurs types de **modèles de régression** afin de prédire la consommation énergétique à partir des différentes sources de production.

Parmi l’ensemble des modèles évalués, deux se sont distingués par leurs performances :

- **Random Forest Regressor**  
  Utilisé pour prédire spécifiquement la consommation sur l’année **2019**.  
  Ce modèle a été retenu pour sa **robustesse**, sa capacité à modéliser des **relations non linéaires**, ainsi que pour sa **bonne interprétabilité**.

- **XGBoost**  
  Utilisé pour prédire la consommation sur **l’ensemble des années** du dataset.  
  Ce modèle a montré d’excellentes performances en matière de **précision** et de **généralisation**.

Ces deux modèles ont obtenu les **meilleurs résultats** lors de nos tests comparatifs.
""")
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel
    st.markdown("""
        <h1 style="text-align: center; color: #1E3A8A; font-size: 36px;">
            📊 RandomForest - Comparaison entre consommation réelle et prédite (2019)
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("### Paramètres du modèle Random Forest")
    st.markdown("""
            - **Modèle** : RandomForestRegressor
            - **Nombre d'arbres (n_estimators)** : 200  
            - **Profondeur maximale des arbres (max_depth)** : 20  
            - **Taille du jeu de test** : 20 %  
            - **Random State** : 42
            """)
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
        
            df_filtered["Mois"] = df_filtered["Date - Heure"].dt.to_period("M")
            df_grouped = df_filtered.groupby("Mois")[["Consommation (MW)", "Consommation Prédite"]].mean().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_grouped["Mois"].astype(str), df_grouped["Consommation (MW)"], width=0.4, label="Consommation réelle", color="#3B528B", align='center')
            ax.bar(df_grouped["Mois"].astype(str), df_grouped["Consommation Prédite"], width=0.4, label="Consommation prédite", color="#84CA66", align='edge')
            ax.set_xlabel("Mois")
            ax.set_ylabel("Consommation moyenne (MW)")
            ax.set_title("Comparaison entre consommation réelle et prédite par mois (2019)")
            ax.legend(loc="upper right")
            plt.xticks(rotation=45)
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            st.write("")  # Espace visuel
            st.write("")  # Espace visuel

            #affichage feactures importances Randomforest
            st.subheader("Importance des Variables")
            feature_importance_df = pd.read_csv("feature_importance_global.csv")
            fig, ax = plt.subplots(figsize=(10, 6))
    
            sns.barplot(data= feature_importance_df, x='Importance', y='Variable',palette="viridis", ax=ax)
            ax.set_title("Importance des variables dans le modèle Random Forest")
            st.pyplot(fig)

            # 📌 Affichage des métriques
            df_eval = df_filtered[["Consommation (MW)", "Consommation Prédite"]].dropna()
            mse = mean_squared_error(df_eval["Consommation (MW)"], df_eval["Consommation Prédite"])
            r2 = r2_score(df_eval["Consommation (MW)"], df_eval["Consommation Prédite"])
            st.write("")  # Espace visuel
            st.write("")  # Espace visuel
            st.markdown("### Résultats de l'entraînement")
            st.write(f"📉 **Erreur quadratique moyenne (MSE) :** {mse:.2f}")
            st.write(f"📈 **Score R² :** {r2:.2f}")
            st.write("")  # Espace visuel
            st.write("")  # Espace visuel
           

    if df is not None:
        predictions_XGBOOST = "predictionsXGBOOST.csv"

        # 📌 Préparer les données
        X_production = df[["Thermique (MW)", "Nucléaire (MW)", "Eolien (MW)", "Solaire (MW)", 
                           "Hydraulique (MW)", "Bioénergies (MW)", "Pompage (MW)"]]
        X_production = X_production.replace(['ND', '-'], 0).fillna(0)

        # 📌 Chargement des prédictions XGBOOST
        if os.path.exists(predictions_XGBOOST):
            df_filtered2 = pd.read_csv(predictions_XGBOOST, parse_dates=["Date - Heure"])
        else:
            st.error("Le fichier de prédictions XGBOOST n'a pas été trouvé.")
            df_filtered2 = pd.DataFrame()
        
        # 📌 Affichage des résultats
        if not df_filtered2.empty:
            st.markdown("""
        <h1 style="text-align: center; color: #1E3A8A; font-size: 36px;">
            📊 XGBoost - Comparaison entre consommation réelle et prédite par année
        </h1>
    """, unsafe_allow_html=True)
             
                                                                                                                                                
                                                                                                                                            
        # 📌 Définition du modèle
            st.markdown("### Paramètres du modèle XGBoost")
            st.write("**Modèle :** XGBoost")
            st.write("**Nombre d'arbres (n_estimators) :** 100")
            st.write("**Learning rate :** 0.1")
            st.write("**Taille du jeu de test :** 20%")
            st.write("**Random State :** 42")

            df_filtered2["Année"] = df_filtered2["Date - Heure"].dt.to_period("Y")
            df_grouped2 = df_filtered2.groupby("Année")[["Consommation (MW)", "Consommation Prédite (MW)"]].mean().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_grouped2["Année"].astype(str), df_grouped2["Consommation (MW)"], width=0.4, label="Consommation réelle", color="#3B528B", align='center')
            ax.bar(df_grouped2["Année"].astype(str), df_grouped2["Consommation Prédite (MW)"], width=0.4, label="Consommation prédite", color="#84CA66", align='edge')
            ax.set_xlabel("Année")
            ax.set_ylabel("Consommation moyenne (MW)")
            ax.set_title("Comparaison entre consommation réelle et prédite par année")
            ax.legend(loc="upper right")
            plt.xticks(rotation=45)
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        # 📌 Affichage des métriques
            df_eval2 = df_filtered2[["Consommation (MW)", "Consommation Prédite (MW)"]].dropna()
            mse = mean_squared_error(df_eval2["Consommation (MW)"], df_eval2["Consommation Prédite (MW)"])
            r2 = r2_score(df_eval2["Consommation (MW)"], df_eval2["Consommation Prédite (MW)"])
            st.write("## Résultats de l'entraînement")
            st.write(f"📉 **Erreur quadratique moyenne (MSE) :** {mse:.2f}")
            st.write(f"📈 **Score R² :** {r2:.2f}")
            st.write("")  # Espace visuel
            st.write("")  # Espace visuel

         # 📌 Intégration du fichier des "features importance"
            fichier_importance_XGBOOST_csv = "feature_importances_XGBOOST.csv"

            try:
                df_importance = pd.read_csv(fichier_importance_XGBOOST_csv)
            except FileNotFoundError:
                st.error(f"Erreur : Le fichier '{fichier_importance_XGBOOST_csv}' n'a pas été trouvé. ")
                st.stop() # Arrête l'exécution si le fichier n'est pas là
            except Exception as e:
                st.error(f"Une erreur est survenue lors du chargement du fichier d'importances : {e}")
                st.stop()

        # 📌 Création du graphique "features importance"
            st.subheader("Importance des Variables")

            fig, ax = plt.subplots(figsize=(10, 7)) 
            sns.barplot(x='Importance', y='Feature', data=df_importance, ax=ax, palette='viridis')
            ax.set_title("Influence des variables sur la prédiction de la consommation")
            ax.set_xlabel("Score d'Importance")
            ax.set_ylabel("Variable")
            plt.tight_layout()

            st.pyplot(fig)

            st.write("")  # Espace visuel
            st.write("")  # Espace visuel
            st.write("")  # Espace visuel
            st.write("")  # Espace visuel
            

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
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel
    st.markdown("""Afin de **dépasser les attentes du projet initial**, nous avons choisi d'aller plus loin en explorant des **modèles de séries temporelles** pour projeter la consommation énergétique **jusqu'en 2030**.

Nous avons d'abord testé des modèles comme **Prophet** et un modèle hybride **Prophet + ARIMA**.  
S'ils prédisaient avec une grande précision les **données déjà présentes dans le dataset**, ils se sont révélés **peu performants pour anticiper la consommation future**, notamment sur un horizon long terme.

Nous nous sommes donc tournés vers un modèle **XGBoost**, en enrichissant le dataset avec des **variables temporelles ingénierées** :
- **Mois** et **année**,  
- **Lags** à 12 et 24 mois,  
- **Moyennes mobiles** (24 mois et différentiel 24–12),  
- **Composantes saisonnières** (**sinus** et **cosinus**).

Ce modèle parvient à bien capturer les **pics saisonniers récurrents**, mais **échoue à refléter les tendances de fond sur le long terme** : il projette une consommation **quasi linéaire**, sans tenir compte d'une probable **hausse structurelle de la demande énergétique**, ce qui est irréaliste dans le contexte français actuel.

Il n’est donc **pas encore prêt pour une mise en production**, car il nécessiterait l’ajout de **variables exogènes** (telles que des données météo, économiques ou réglementaires) pour mieux expliquer l’évolution de la consommation et **améliorer le R²**.
                            """)
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel


    if df_pred is not None:
        # 📌 Sélection d'une année (2024-2030)
        year_selected = st.selectbox("📅 Sélectionnez une année :", list(range(2024, 2031)))

        # 📌 Sélection d'un mois
        month_selected = st.selectbox("📆 Sélectionnez un mois :", list(range(1, 13)))
        st.write("")  # Espace visuel

        # 📌 Filtrer les prédictions pour le mois et l'année sélectionnés
        filtered_df = df_pred[
            (df_pred['date'].dt.year == year_selected) & 
            (df_pred['date'].dt.month == month_selected)
        ]
        st.write("")  # Espace visuel

        # 📌 Affichage de la prévision
        if not filtered_df.empty:
            pred_value = filtered_df['xgboost'].values[0]
            st.metric(label=f"Prédiction XGBoost pour {month_selected}/{year_selected}", value=f"{pred_value:.2f} MW")
        else:
            st.warning("⚠️ Aucune donnée disponible pour ce mois/année.")
            st.write("")  # Espace visuel

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
       
elif page == pages[4]:
    st.write("## 🔚 Conclusion générale")
    st.write("")  # Espace visuel
    st.write("")  # Espace visuel
    st.markdown("""
## Conclusion & Perspectives

Ce projet nous a permis d'explorer plusieurs approches de modélisation pour anticiper la consommation énergétique en France à partir des données de production. Grâce à une analyse exploratoire rigoureuse et une sélection fine des variables, nous avons pu construire un modèle robuste, mais encore perfectible.

### ✅ Points forts du projet
- Déploiement d’un **modèle XGBOOST** et d’un modèle **RandomForestRegressor**, atteignant jusqu’à **95 % de précision** sur les données historiques.
- Intégration dans une **application Streamlit interactive**, facilitant l'exploration des prédictions et des variables.

### ⛔ Contraintes du projet
- Manque de temps pour approfondir certaines analyses et intégrer plus de données (vent, ensoleillement).
- Gestion des obligations personnelles et professionnelles en parallèle du projet.

### ⚙️ Limites techniques
- Difficultés liées à la performance de Google Colab avec des datasets volumineux.
- Manque de ressources pour certains modèles.
- Accès limité à des données météo ou contextuelles (ex : COVID, maintenance, INSEE).

### 📈 Résultats de modélisation
- Le modèle permet de mieux anticiper la consommation, détecter des tensions réseau ou des risques de blackout.
- Il contribue à la prise de décisions pour des actions préventives (délestage, flexibilité).
- Le modèle XGBoost s’est révélé le plus performant avec une très bonne précision, mais les résultats sont limités sur le long terme en raison du manque de variables explicatives comme la température.

### 🔍 Analyse des résultats
- La production énergétique est stable (notamment nucléaire), avec des disparités régionales.
- La consommation varie fortement selon les saisons (pic hivernal) et les conditions météo extrêmes.
- L’équilibrage consommation-production doit être sécurisé par des flux d’énergie venant des autres pays ou marchés.

### 🚀 Applications futures
- Utilisation du modèle pour anticiper les échanges d’énergie avec les pays voisins.
- Optimisation des achats sur les marchés spot (J+1, H+1).
- Ajustement en temps réel de l’équilibre offre-demande.
- Ajout de **facteurs exogènes** pour améliorer la précision.
- **Tester des modèles LSTM** pour mieux détecter les tendances longues et les ruptures.
- Intégrer un **système d’alerte basé sur des seuils critiques**.
- **Mise à jour continue** du modèle avec les nouvelles données mensuelles.
- **Création d’une API ou dashboard web** connecté aux données énergétiques nationales.

---

💡 Ce projet a été une **expérience collective riche**, mêlant modélisation, visualisation et déploiement.  
L'application constitue une première étape vers une **plateforme prédictive plus complète**, au service de la **planification énergétique et de la sécurité du réseau**.
""")
