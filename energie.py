import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image  # Ajout de l'import manquant pour Image
from pathlib import Path

# Afficher le répertoire de travail actuel
current_dir = Path.cwd()
print(f"Répertoire actuel : {current_dir}")

# Lister tous les fichiers dans le répertoire
print("\nFichiers dans le répertoire :")
for file in current_dir.glob('*'):
    print(file.name)

@st.cache_data
def get_logo():
    file_id = "1ZF4CX_g41jhOjNipe9OhCTB7mnDLn6Ed"
    url = f"https://drive.google.com/file/d/15l7StwyKMtW9dGB-MrnD_hcZtELMCqbz/view?usp=drive_link={file_id}"
    response = requests.get(url)
    return BytesIO(response.content)

# Chargement des données
current_dir = Path(__file__).parent
csv_path = current_dir / "eco2mix-regional-cons-def.csv"
print(f"Le fichier existe : {csv_path.exists()}")
print(f"Chemin complet : {csv_path}")

# Lecture du CSV
df = pd.read_csv(csv_path, sep=';')  # Ajout de la lecture du DataFrame

# Création du DataFrame de comparaison
comparaison = pd.DataFrame({
    'Production Totale': df['Production_Totale'],
    'Consommation': df['Consommation (MW)']
})

# Pour utiliser resample, il faut d'abord définir un index temporel
# Assurez-vous d'avoir une colonne de date et définissez-la comme index
# Par exemple (ajustez le nom de la colonne selon votre CSV) :
# df['Date'] = pd.to_datetime(df['Date'])
# comparaison.index = df['Date']

# Calcul des moyennes annuelles
# comparaison_annuelle = comparaison.resample('Y').mean()

# Interface Streamlit
st.title("Observatoire de la production et consommation électrique en France ")

# Affichage du logo dans la sidebar
col1, col2, col3 = st.columns([1,2,1])
with col1:
    try:
        logo = Image.open(Path(current_dir, "logo apply.png"))
        st.image(logo, width=200)
    except FileNotFoundError:
        st.error("Logo non trouvé")

# Sidebar
st.sidebar.title("Sommaire")
pages = ["Introduction", "Exploration du jeu de données", "Statistiques et indicateurs", "Modélisation"]
page = st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
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




