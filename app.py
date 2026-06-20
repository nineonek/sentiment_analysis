import streamlit as st
import pandas as pd
import sqlite3

# Set page layout to wide
st.set_page_config(layout="wide")

# 1. Load the dataset (cached so it's instant after the first load)
@st.cache_data
def load_project_data():
    # If using CSV: return pd.read_csv("app_data.csv", parse_dates=['date'])
    df = pd.read_parquet("app_data.parquet")
    df['date'] = pd.to_datetime(df['date'])
    return df

try:
    df_clean = load_project_data()
except Exception as e:
    st.error("Data file 'app_data.parquet' not found. Make sure to upload it to GitHub!")
    st.stop()

# 2. Local SQLite Database to track what your professor searches
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

# 3. Streamlit Interface
st.title("🔎 Media Database Keyword Filter")
st.write("Search for terms across the entire database to analyze real-time sentiment distribution.")

# Input layout
keyword = st.text_input("Enter a keyword to query the article database (e.g., 'Lufthansa', 'Suicide', 'Search'):")

if keyword:
    # Filter dataset using case-insensitive string matching across headline or title
    mask = df_clean['headline'].str.contains(keyword, case=False, na=False)
    if 'title' in df_clean.columns:
        mask = mask | df_clean['title'].str.contains(keyword, case=False, na=False)
        
    results = df_clean[mask]
    
    if not results.empty:
        avg_score = float(results['sentiment_score'].mean())
        count = len(results)
        
        # Save search trace to local SQLite tracking table
        cursor.execute("INSERT INTO history (keyword, matches_found, avg_sentiment) VALUES (?, ?, ?)", 
                       (keyword, count, avg_score))
        conn.commit()
        
        # Display Metrics side by side
        col1, col2 = st.columns(2)
        col1.metric("Articles Found", f"{count}")
        col2.metric("Average Sentiment Score", f"{avg_score:.4f}", help="Scale goes from -1 (Negative) to +1 (Positive)")
        
        # Show matching articles table
        st.subheader(f"Matching entries for '{keyword}'")
        st.dataframe(results[['date', 'headline', 'sentiment_score']].sort_values(by='date', ascending=False), use_container_width=True)
    else:
        st.warning(f"No articles found matching the term '{keyword}'. Try another word!")

# 4. Show Saved Logs Database tracking at the bottom
st.write("---")
st.subheader("📊 Professor's Search History Log (SQLite)")
df_history = pd.read_sql_query("SELECT id, keyword, matches_found, avg_sentiment FROM history ORDER BY id DESC", conn)

if not df_history.empty:
    st.dataframe(df_history, use_container_width=True)
    if st.button("Clear Search Logs"):
        cursor.execute("DELETE FROM history")
        conn.commit()
        st.rerun()
else:
    st.info("The query log database is currently empty.")