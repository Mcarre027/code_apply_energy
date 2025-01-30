import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path

# Fonction pour charger le CSV avec mise en cache
@st.cache_data
def load_data():
    """Charge le fichier CSV en mémoire avec gestion d'erreur."""
    current_dir = Path.cwd()
    csv_path = current_dir / "eco2mix-regional-cons-def.csv"

    # Debug : afficher les informations sur le fichier
    print(f"Chemin complet du CSV : {csv_path}")
    print(f"Le fichier existe : {csv_path.exists()}")

    try:
        df = pd.read_csv(csv_path, sep=';', low_memory=False)
        print("Lecture du CSV réussie !")
        return df
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV : {e}")
        st.error(f"Erreur lors de la lecture du CSV : {e}")
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
        st.write("#### **Dataset eco2mix régional**")
        st.dataframe(df.head(10))
        
        st.write("#### **Dimension du dataset**")
        st.write(df.shape)

        st.write("#### **Statistiques des variables du dataset**")
        st.dataframe(df.describe())

        st.write("#### **Affichage des valeurs manquantes**")
        if st.checkbox("Afficher les NA"):
            st.dataframe(df.isna().sum())
    else:
        st.error("Le dataset n'a pas pu être chargé.")

# Page Statistiques et indicateurs
elif page == pages[2]:
    st.write("### 📊 Statistiques et indicateurs")

    if df is not None:
        # Préparation des colonnes de production et conversion en numérique
        colonnes_production = ["Thermique (MW)", "Nucléaire (MW)", "Eolien (MW)", "Solaire (MW)", 
                               "Hydraulique (MW)", "Bioénergies (MW)"]

        for col in colonnes_production:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calcul de la production totale directement sur le DataFrame principal
        df["Production_Totale"] = df[colonnes_production].sum(axis=1, skipna=True)

        # Création de la variable comparaison
        df_clean = df.copy()

        comparaison = pd.DataFrame({
            "Production Totale": df_clean["Production_Totale"],
            "Consommation": df_clean["Consommation (MW)"]
        })

        # Conversion de la colonne "Date - Heure" en datetime
        df_clean["Date - Heure"] = pd.to_datetime(df_clean["Date - Heure"], errors="coerce")

        # Définir l'index sur la date
        comparaison.index = df_clean["Date - Heure"]

        # Agrégation annuelle
        comparaison_annuelle = comparaison.resample("Y").mean()

        # Couleurs
        color_production = "blue"
        color_consommation = "orange"

        # Affichage du graphique
        st.write("#### Comparaison de la Production et de la Consommation Totale par Année")
        fig, ax = plt.subplots(figsize=(16, 8))

        comparaison_annuelle.plot(
            kind="bar",
            color=[color_production, color_consommation],
            width=0.8,
            ax=ax
        )

        ax.set_title("Comparaison Production vs Consommation Totale par Année",
                     fontsize=16,
                     pad=20)
        ax.set_xlabel("Année", fontsize=12)
        ax.set_ylabel("Moyenne annuelle (MW)", fontsize=12)
        ax.set_xticklabels([label.get_text()[:4] for label in ax.get_xticklabels()],
                           rotation=45, ha="right")
        ax.legend(["Production Totale", "Consommation"], loc="upper left", fontsize=12)
        ax.grid(True, alpha=0.2, axis="y")

        # Affichage dans Streamlit
        st.pyplot(fig)

        # Affichage des statistiques
        st.write("#### Statistiques annuelles")
        st.dataframe(comparaison_annuelle.describe())

        # Matrice de corrélation
        st.write("#### Matrice de corrélation entre consommation et sources de production")
        
        colonnes_analyse = ["Consommation (MW)", "Production_Totale", "Thermique (MW)",
                            "Nucléaire (MW)", "Eolien (MW)", "Solaire (MW)",
                            "Hydraulique (MW)", "Bioénergies (MW)", "Ech. physiques (MW)", "Pompage (MW)"]

        correlation_matrix = df_clean[colonnes_analyse].corr()

        # Création et affichage de la heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title("Matrice de corrélation entre consommation et sources de production")
        st.pyplot(fig)

        # Comparaison Production vs Consommation par région
        st.write("#### Comparaison Production vs Consommation par région")
        production_par_region = df.groupby('Région')["Production_Totale"].sum()
        consommation_par_region = df.groupby('Région')["Consommation (MW)"].sum()

        comparaison_region = pd.DataFrame({
            'Production': production_par_region,
            'Consommation': consommation_par_region
        })
        
        fig, ax = plt.subplots(figsize=(24, 12))
        comparaison_region.plot(kind='bar', ax=ax)
        ax.set_title('Comparaison Production vs Consommation par région (MW)', fontsize=16)
        ax.set_xlabel('Région', fontsize=12)
        ax.set_ylabel('MW', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Graphique empilé de la répartition des sources de production par région
        st.write("#### Répartition des sources de production par région")
        colonnes_production = ['Thermique (MW)', 'Nucléaire (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)', 'Bioénergies (MW)', 'Pompage (MW)']
        production_detail = df.groupby('Région')[colonnes_production].sum()
        fig = go.Figure()

        for colonne in colonnes_production:
            fig.add_trace(go.Bar(
                name=colonne.replace(' (MW)', ''),
                x=production_detail.index,
                y=production_detail[colonne],
                hovertemplate="Région: %{x}<br>" +
                             f"{colonne}: %{{y:,.0f}} MW<br>" +
                             "<extra></extra>"))

        fig.update_layout(
            barmode='stack',
            title='Répartition des sources de production par région (MW)',
            title_x=0.5,
            xaxis_title='Région',
            yaxis_title='Production (MW)',
            width=1500,
            height=800,
            showlegend=True,
            legend=dict(
                x=1.05,
                y=1,
                xanchor='left'
            ),
            xaxis=dict(
                tickangle=45,
                tickmode='array',
                ticktext=production_detail.index,
                tickvals=list(range(len(production_detail.index)))
            ),
            yaxis=dict(
                gridcolor='LightGrey',
                gridwidth=0.5,
            )
        )

        fig.update_traces(marker_line_color='rgb(8,48,107)',
                         marker_line_width=1.5,
                         opacity=0.85)

        st.plotly_chart(fig)
        # Création des variables renouvelables et non renouvelables
        df['NonRenew'] = df[['Thermique (MW)', 'Nucléaire (MW)']].sum(axis=1)
        df['Renew'] = df[['Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)', 'Bioénergies (MW)']].sum(axis=1)

        # Conversion de la colonne 'Date - Heure' au format datetime si ce n'est pas déjà fait
        df['Date - Heure'] = pd.to_datetime(df['Date - Heure'], errors='coerce')

        # Création de la colonne 'Année'
        df['Année'] = df['Date - Heure'].dt.year

        # Filtrage des données pour la période 2015-2022
        df15_22 = df[(df['Année'] >= 2015) & (df['Année'] <= 2022)]

        # Affichage du titre
        st.write("#### Analyse des dynamiques de production entre les énergies renouvelables et non renouvelables (2015-2022)")

        # Affichage du graphique
        fig, ax = plt.subplots(figsize=(10, 6))

        # Tracés des lignes pour les énergies renouvelables et non renouvelables
        sns.lineplot(x='Année', y='Renew', data=df15_22, marker='o', label='Renouvelable', ax=ax)
        sns.lineplot(x='Année', y='NonRenew', data=df15_22, marker='o', label='Non Renouvelable', ax=ax)

        # Configuration des axes et du titre
        ax.set_title("Évolution de la production d'énergie par année")
        ax.set_xlabel("Année")
        ax.set_ylabel("Production (MW)")
        ax.legend(title="Type d'énergie")

        # Affichage du graphique dans Streamlit
        st.pyplot(fig)

# Page Modélisation
elif page == pages[3]:
    st.write("### 🤖 Modélisation")
