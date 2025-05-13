README.md

🔋 Application Streamlit : Prédiction de la Consommation électrique en France

Cette application Streamlit permet de prédire et d’analyser la consommation électrique française à partir des données de production régionales (Thermique, Nucléaire, Renouvelables, etc.).
Elle a été conçue dans le cadre d’un projet de formation Data Analyst.

🚀 Objectifs de l'application

🔍 Explorer les données de consommation énergétique françaises.

🧠 Visualiser les tendances de production et leur impact sur la consommation.

🤖 Prédire la consommation électrique avec un modèle Random Forest.

🔢 Projeter les consommations mensuelles jusqu'en 2030 avec XGBoost.

🔧 Fonctionnalités

📈 Exploration des données : aperçu, statistiques descriptives, valeurs manquantes

🔍 Analyse graphique : comparaison consommation vs production, corrélations

📊 Justification des variables utilisées pour la prédiction

🤞 Modélisation Random Forest : affichage MSE, R², importance des features

🔍 Comparaison réelle/prédite sur 2019

🔮 Projection XGBoost jusqu'en 2030 avec filtre dynamique par année/mois

🚀 Modèles utilisés

RandomForestRegressor (Sklearn)

n_estimators = 200

max_depth = 20

test_size = 20%

r² moyen sur historique ≈ 0.95

XGBoost (projection 2030)

Séries temporelles Prophet et ARIMA (testés en phase exploratoire)

🌐 Aperçu de l'application

Graphique : comparaison entre consommation réelle et prédite sur 2019

📁 Structure du projet

.
├── app.py                         # Application principale Streamlit
├── eco2mix-regional-cons-def.csv # Données d'origine (RTE)
├── predictions_2019.csv          # Résultats sur 2019
├── predictions_2030.csv          # Prédictions futures XGBoost
├── feature_importance_global.csv # Importance des variables Random Forest
├── style.css                     # Feuille de style (optionnelle)
├── requirements.txt              # Packages requis
└── README.md                     # Ce fichier

⚙️ Installation locale

1. Créer un environnement virtuel

python -m venv env
source env/bin/activate  # ou env\Scripts\activate sur Windows

2. Installer les dépendances

pip install -r requirements.txt

3. Lancer l'application

streamlit run app.py

📈 Données utilisées

Les données proviennent du jeu Eco2Mix Régional publié par RTE France.

Colonnes utilisées :

Date - Heure

Consommation (MW)

Thermique, Nucléaire, Eolien, Solaire, Hydraulique, Bioénergies, Pompage

🚨 Limites et pistes d'amélioration

❌ Manque de données exogènes (température, événements)

🌀 Les modèles actuels capturent mal les ruptures de tendance

⚖️ Prochaine étape : intégration de LSTM et mécanismes d'alerte pré-blackout

🤝 Auteurs

Projet réalisé par [Votre Nom / Équipe] dans le cadre de la formation Data Analyst RNCP7 - DataScientest x École des Mines ParisTech.

📄 Licence

Usage libre à des fins pédagogiques et de démonstration.

