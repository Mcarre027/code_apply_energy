# 🔋 Application Streamlit : Prédiction de la Consommation électrique en France

Cette application Streamlit permet de prédire et d’analyser la consommation électrique française à partir des données de production régionales (Thermique, Nucléaire, Renouvelables, etc.). Elle a été conçue dans le cadre d’un projet de formation Data Analyst.

---

## 🚀 Objectifs de l'application

* Explorer les données de consommation énergétique françaises
* Visualiser les tendances de production et leur impact sur la consommation
* Prédire la consommation électrique avec un modèle Random Forest
* Projeter les consommations mensuelles jusqu'en **2030** avec XGBoost

---

## 🔧 Fonctionnalités

* Exploration des données : aperçu, statistiques descriptives, valeurs manquantes
* Analyse graphique : comparaison consommation vs production, corrélations
* Justification des variables utilisées pour la prédiction
* Modélisation Random Forest : affichage MSE, R², importance des variables
* Comparaison réelle/prédite sur 2019
* Projection XGBoost jusqu'en 2030 avec filtre dynamique par année et mois

---

## 🚀 Modèles utilisés

* `RandomForestRegressor` (Sklearn)

  * `n_estimators=200`
  * `max_depth=20`
  * `test_size=20%`
  * `random_state=42`
  * Score R² moyen ≈ 0.95
* `XGBoost` pour les projections à long terme (2030)
* Prophet / ARIMA utilisés en phase exploratoire

---

## 📁 Structure du projet

```
.
├── energie.py                    # Application principale Streamlit
├── eco2mix-regional-cons-def.csv # Données d'origine (RTE)
├── predictions_2019.csv          # Prédictions vs réel sur 2019
├── predictions_2030.csv          # Prédictions futures (XGBoost)
├── feature_importance_global.csv # Importance des variables
├── style.css                     # Feuille de style optionnelle
├── requirements.txt              # Packages requis
└── README.md                     # Ce fichier
```

---

## ⚙️ Installation locale

1. Créer un environnement virtuel :

```bash
python -m venv env
source env/bin/activate  # ou env\Scripts\activate pour Windows
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

3. Lancer l'application :

```bash
streamlit run app.py
```

---

## 📈 Données utilisées

Les données proviennent du jeu **Eco2Mix Régional** publié par **RTE France**.

Colonnes principales :

* Date - Heure
* Consommation (MW)
* Thermique, Nucléaire, Eolien, Solaire, Hydraulique, Bioénergies, Pompage

---

## 🚨 Limites et pistes d'amélioration

* Absence de données exogènes (météo, événements, jours fériés)
* Faible capture des ruptures de tendance longue
* Intégration future d'algorithmes LSTM et de mécanismes d'alerte

---

## 🤝 Auteurs

Projet réalisé par **\[CARRE Matthieu & AUDIBERT Jérémy / équipe oct24_cda_energie ]** dans le cadre de la formation **Data Analyst RNCP7** - DataScientest x École des Mines ParisTech

---

## 📄 Licence

Ce projet est mis à disposition à des fins pédagogiques et de démonstration. Libre d’utilisation et d’amélioration avec mention de la source.

