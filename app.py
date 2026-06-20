import streamlit as st
import pandas as pd
import sqlite3
from transformers import pipeline

# Config de la page
st.set_page_config(layout="wide")

# 1. Chargement de la base de données de base
@st.cache_data
def load_project_data():
    df = pd.read_parquet("app_data.parquet")
    df['date'] = pd.to_datetime(df['date'])
    # On s'assure que tout est au format texte pour éviter les bugs de recherche
    df['headline'] = df['headline'].fillna('').astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    return df

try:
    df_clean = load_project_data()
except Exception as e:
    st.error("Fichier 'app_data.parquet' introuvable. Charge-le sur GitHub !")
    st.stop()

# 2. Chargement du modèle d'IA (mis en cache pour ne pas ramer)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

analyzer = load_sentiment_model()

# Fonction maison pour calculer le score (la même que dans ton notebook)
def compute_sentiment(text):
    if not text.strip():
        return 0.0
    res = analyzer(text[:512])[0]
    score = res['score']
    return -score if res['label'] == 'NEGATIVE' else score

# 3. Base de données locale SQLite pour l'historique du prof
conn = sqlite3.connect("search_history.db", check_same_thread=False)
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
st.write("Entre un mot-clé : l'IA va filtrer les articles et calculer leur score de sentiment en direct.")

keyword = st.text_input("Mot-clé à chercher (ex: Lufthansa, Suicide, Search, Paris) :")

if keyword:
    # Filtrage par mot-clé (dans headline OU title)
    mask = df_clean['headline'].str.contains(keyword, case=False, na=False) | \
           df_clean['title'].str.contains(keyword, case=False, na=False)
        
    results = df_clean[mask].copy()
    
    if not results.empty:
        # On affiche un spinner le temps que l'IA bosse
        with st.spinner(f"Calcul du sentiment pour {len(results)} articles..."):
            # On applique le modèle sur la colonne headline pour le calcul en direct
            results['sentiment_score'] = results['headline'].apply(compute_sentiment)
        
        avg_score = float(results['sentiment_score'].mean())
        count = len(results)
        
        # Sauvegarde dans l'historique SQLite
        cursor.execute("INSERT INTO history (keyword, matches_found, avg_sentiment) VALUES (?, ?, ?)", 
                       (keyword, count, avg_score))
        conn.commit()
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        col1.metric("Articles trouvés", f"{count}")
        col2.metric("Moyenne Emotionnelle (Sentiment)", f"{avg_score:.4f}", help="De -1 (Ultra-négatif) à +1 (Positif/Neutre)")
        
        st.subheader(f"Articles correspondants à '{keyword}'")
        st.dataframe(results[['date', 'headline', 'sentiment_score']].sort_values(by='date', ascending=False), use_container_width=True)
    else:
        st.warning(f"Aucun article ne contient le mot '{keyword}'.")

# 5. Historique des requêtes tout en bas
st.write("---")
st.subheader("📊 Historique des recherches de la session (Base SQLite)")
df_history = pd.read_sql_query("SELECT id, keyword, matches_found, avg_sentiment FROM history ORDER BY id DESC", conn)

if not df_history.empty:
    st.dataframe(df_history, use_container_width=True)
    if st.button("Effacer l'historique"):
        cursor.execute("DELETE FROM history")
        conn.commit()
        st.rerun()