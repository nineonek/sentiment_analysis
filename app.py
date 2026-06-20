import streamlit as st
import pandas as pd
import sqlite3
from transformers import pipeline

# Config de la page
st.set_page_config(layout="wide", page_title="Analyseur de Sentiment", page_icon="🔎")

# 1. Chargement de la base de données de base
@st.cache_data
def load_project_data():
    df = pd.read_parquet("app_data.parquet")
    df['date'] = pd.to_datetime(df['date'])
    if 'headline' in df.columns:
        df['headline'] = df['headline'].fillna('').astype(str)
    if 'title' in df.columns:
        df['title'] = df['title'].fillna('').astype(str)
    return df

try:
    df_clean = load_project_data()
except Exception as e:
    st.error("Fichier 'app_data.parquet' introuvable. Charge-le sur GitHub !")
    st.stop()

# 2. Chargement du modèle d'IA (mis en cache)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

analyzer = load_sentiment_model()

def compute_sentiment(text):
    if not text.strip():
        return 0.0
    res = analyzer(text[:512])[0]
    score = res['score']
    # Normalisation : si c'est négatif, on inverse le signe du score de confiance
    return -score if res['label'] == 'NEGATIVE' else score

# 3. Base de données locale SQLite
def get_db_connection():
    return sqlite3.connect("search_history.db", check_same_thread=False)

conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        keyword TEXT,
        matches_found INTEGER,
        avg_sentiment REAL
    )
""")
conn.commit()

# 4. Interface Streamlit
st.title("🔎 Analyseur de Sentiment Global en Temps Réel")
st.write("Filtrez les articles de presse par mot-clé et laissez l'IA évaluer leur charge émotionnelle en direct.")

# --- SECTION EXPLICATIVE : COMMENT ÇA MARCHE ---
with st.expander("ℹ️ Méthodologie : Comment est calculé le sentiment ?", expanded=True):
    st.write("""
    Cette application utilise le modèle de Deep Learning **DistilBERT** (un modèle de traitement du langage naturel dérivé de BERT, optimisé pour être léger et rapide). 
    
    Chaque fois qu'un mot-clé est cherché, l'algorithme analyse la structure textuelle de chaque titre d'article trouvé et extrait deux informations :
    1. **Une étiquette (Label)** : `POSITIVE` (Neutre/Positif) ou `NEGATIVE` (Triste, Alarmiste, Anxiogène).
    2. **Un score de confiance** : Une probabilité variant de `0.5` à `1.0` décrivant la certitude du modèle.
    """)
    
    # Création du tableau explicatif
    df_explain = pd.DataFrame({
        "Plage de Score": ["De -1.0 à -0.5", "Autour de 0.0", "De +0.5 à +1.0"],
        "Signification": ["Titre purement négatif / tragique", "Titre neutre ou ambigu", "Titre factuel, neutre-positif ou optimiste"],
        "Exemple de titre de presse": [
            "Terrorist attack kills dozens in violent clash",
            "Aviation authority opens standard inquiry into procedures",
            "Rescue teams safely locate survivors after long search"
        ]
    })
    st.table(df_explain)
    st.write("**Formule mathématique de normalisation :** Si le verdict est négatif, le score devient $-1 \\times \\text{score de confiance}$. S'il est positif, il reste tel quel. La moyenne globale donne l'indice de cadrage émotionnel du mot-clé.")

# --- Suggestions de recherche rapides ---
st.write("---")
st.write("**Suggestions de recherche rapides :**")
col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
suggested_word = None

if col_ex1.button("✈️ Germanwings"): suggested_word = "Germanwings"
if col_ex2.button("🌏 AirAsia"): suggested_word = "AirAsia"
if col_ex3.button("✏️ Charlie Hebdo"): suggested_word = "Charlie"
if col_ex4.button("🇳🇬 Boko Haram"): suggested_word = "Boko"

if suggested_word:
    keyword = st.text_input("Mot-clé à chercher :", value=suggested_word)
else:
    keyword = st.text_input("Mot-clé à chercher (ex: Lufthansa, Crash, Search, Paris) :")

if keyword:
    mask = df_clean['headline'].str.contains(keyword, case=False, na=False)
    if 'title' in df_clean.columns:
        mask = mask | df_clean['title'].str.contains(keyword, case=False, na=False)
        
    results = df_clean[mask].copy()
    
    if not results.empty:
        with st.spinner(f"Calcul du sentiment via DistilBERT pour {len(results)} articles..."):
            results['sentiment_score'] = results['headline'].apply(compute_sentiment)
        
        avg_score = float(results['sentiment_score'].mean())
        count = len(results)
        
        cursor.execute("INSERT INTO history (keyword, matches_found, avg_sentiment) VALUES (?, ?, ?)", 
                       (keyword, count, avg_score))
        conn.commit()
        
        col1, col2 = st.columns(2)
        col1.metric("Articles trouvés", f"{count}")
        col2.metric("Moyenne Émotionnelle (Sentiment)", f"{avg_score:.4f}", help="De -1 (Ultra-négatif) à +1 (Positif/Neutre)")
        
        st.write("### 📊 Distribution des scores pour cette recherche")
        st.bar_chart(results['sentiment_score'], x_label="Score de sentiment", y_label="Nombre d'articles")
        
        st.subheader(f"📄 Liste des articles correspondants à '{keyword}'")
        colonnes_affichage = ['date', 'headline', 'sentiment_score']
        st.dataframe(results[colonnes_affichage].sort_values(by='date', ascending=False), use_container_width=True)
    else:
        st.warning(f"Aucun article ne contient le mot '{keyword}'.")

# 5. Historique des requêtes tout en bas
st.write("---")
st.subheader("📊 Historique des recherches de la session (Base SQLite)")

df_history = pd.read_sql_query("SELECT id, keyword, matches_found, avg_sentiment FROM history ORDER BY id DESC", conn)

if not df_history.empty:
    st.dataframe(df_history, use_container_width=True)
    
    if st.button("🗑️ Effacer l'historique complet"):
        clear_conn = get_db_connection()
        clear_cursor = clear_conn.cursor()
        clear_cursor.execute("DELETE FROM history")
        clear_conn.commit()
        clear_conn.close()
        st.success("Historique supprimé avec succès !")
        st.rerun()
else:
    st.info("L'historique des requêtes est actuellement vide.")