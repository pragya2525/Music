#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit


# In[2]:


from pyngrok import conf, installer
installer.install_ngrok(conf.get_default().ngrok_path)


# In[3]:


from pyngrok import ngrok
ngrok.set_auth_token("31KPNKThRlF8Ve4fhErGw15OBhn_6Dp2eZfKvqx94zp1tTRjZ")


# In[4]:


get_ipython().system('pip install -q pandas numpy matplotlib seaborn scikit-learn streamlit pyngrok')


# In[5]:


from IPython.display import Markdown, display
display(Markdown("# SelfMade Music Recommendation System\n"
                 "Project 2: Spotify Songs’ Genre Segmentation\n"
                 "Pre-processing • EDA • Correlation • Clustering • Recommendation • Web App (global via ngrok)"))


# In[6]:


import os, pandas as pd
CSV_FILE = "spotify_dataset.csv"
assert os.path.exists(CSV_FILE), f"Place '{CSV_FILE}' in this folder."
df = pd.read_csv(CSV_FILE)
print("Shape:", df.shape)
df.head()


# In[7]:


import numpy as np

df.drop_duplicates(inplace=True)

text_cols = [
    "track_id","track_name","track_artist","track_album_name",
    "playlist_name","playlist_genre","playlist_subgenre"
]
for c in text_cols:
    if c in df.columns:
        df[c] = df[c].astype(str)
    else:
        print(f"⚠️ Missing text column: {c}")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)

for c in text_cols:
    if c in df.columns:
        df[c].replace(["", "nan", "None", "NaN"], np.nan, inplace=True)
        df[c].fillna("Unknown", inplace=True)

print("After preprocessing:", df.shape)
df[text_cols].head()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")

# Genre distribution (top 12)
if "playlist_genre" in df.columns:
    topg = df["playlist_genre"].value_counts().head(12)
    plt.figure(figsize=(9,5))
    sns.barplot(x=topg.values, y=topg.index)
    plt.title("Top Playlist Genres")
    plt.xlabel("Count"); plt.ylabel("playlist_genre"); plt.tight_layout(); plt.show()

# Subgenre distribution (top 12)
if "playlist_subgenre" in df.columns:
    topsg = df["playlist_subgenre"].value_counts().head(12)
    plt.figure(figsize=(9,5))
    sns.barplot(x=topsg.values, y=topsg.index)
    plt.title("Top Playlist Subgenres")
    plt.xlabel("Count"); plt.ylabel("playlist_subgenre"); plt.tight_layout(); plt.show()

# Popularity distribution
if "track_popularity" in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df["track_popularity"], bins=30, kde=True)
    plt.title("Track Popularity Distribution")
    plt.xlabel("track_popularity"); plt.tight_layout(); plt.show()


# In[9]:


corr_cols = [
    "danceability","energy","key","loudness","mode","speechiness",
    "acousticness","instrumentalness","liveness","valence",
    "tempo","duration_ms","track_popularity"
]
avail = [c for c in corr_cols if c in df.columns]
print("Correlation features:", avail)

plt.figure(figsize=(12,8))
sns.heatmap(df[avail].corr(), cmap="coolwarm", square=True)
plt.title("Correlation Matrix"); plt.tight_layout(); plt.show()


# In[10]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

model_feats = [
    "danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo",
    "duration_ms","track_popularity"
]
features = [f for f in model_feats if f in df.columns]
print("Clustering features:", features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features].fillna(df[features].median()))

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=42)
p2 = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = p2[:,0], p2[:,1]

df["Cluster"].value_counts().sort_index()


# In[11]:


plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=18)
plt.title("Song Clusters (PCA)"); plt.legend(title="Cluster"); plt.tight_layout(); plt.show()


# In[12]:


# Top genres per cluster
if "playlist_genre" in df.columns:
    top_g = (
        df.groupby(["Cluster","playlist_genre"])
          .size().reset_index(name="count")
          .sort_values(["Cluster","count"], ascending=[True, False])
          .groupby("Cluster").head(3)
    )
    print("Top playlist_genre per cluster:")
    display(top_g)

# Top playlists per cluster
if "playlist_name" in df.columns:
    top_p = (
        df.groupby(["Cluster","playlist_name"])
          .size().reset_index(name="count")
          .sort_values(["Cluster","count"], ascending=[True, False])
          .groupby("Cluster").head(3)
    )
    print("Top playlist_name per cluster:")
    display(top_p)


# In[13]:


CLEAN = "spotify_cleaned.csv"
df.to_csv(CLEAN, index=False)
print("Saved:", CLEAN)


# In[14]:


get_ipython().run_cell_magic('writefile', 'spotify_app.py', 'import streamlit as st\nimport pandas as pd\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.neighbors import NearestNeighbors\nimport urllib.parse\n\ndf = pd.read_csv("spotify_cleaned.csv")\n\nst.set_page_config(page_title="SelfMade Music Recommendation System", layout="wide")\nst.markdown("""\n<style>\n/* Page background: bluish gradient */\n[data-testid="stAppViewContainer"] {\n  background: linear-gradient(135deg, #a7d3ff 0%, #1e3c72 100%);\n}\n.block-container {max-width: 1100px;}\n/* white search input */\n.stTextInput > div > div > input {\n  background: #ffffff !important; color: #000 !important;\n  border-radius: 10px; border: 1px solid #ddd;\n}\n/* golden buttons */\n.stButton > button {\n  background: linear-gradient(90deg, #d4af37, #f1c40f);\n  color: #000; font-weight: 700; border-radius: 10px; border: 0;\n  padding: 0.5rem 1rem;\n}\n/* result card */\n.song-card {\n  background: rgba(255,255,255,0.08);\n  border: 1px solid rgba(255,255,255,0.18);\n  border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; color: #fff;\n}\na { color: #ffe082; font-weight: 600; text-decoration: none; }\na:hover { text-decoration: underline; }\n</style>\n""", unsafe_allow_html=True)\n\nst.title("🎵 SelfMade Music Recommendation System")\nst.caption("Project 2: Spotify Songs’ Genre Segmentation — Search · Correlate · Cluster · Recommend")\n\nsearch_field = st.selectbox(\n    "Search by",\n    ["Any","track_id","track_name","track_artist","track_album_name","playlist_name","playlist_genre","playlist_subgenre"]\n)\nquery = st.text_input("Enter your search term (case-insensitive):", "")\n\ndef search_rows(q: str, field: str) -> pd.DataFrame:\n    if not q: return pd.DataFrame()\n    ql = q.lower()\n    if field == "Any":\n        mask = pd.Series(False, index=df.index)\n        for c in ["track_id","track_name","track_artist","track_album_name","playlist_name","playlist_genre","playlist_subgenre"]:\n            if c in df.columns:\n                mask |= df[c].astype(str).str.lower().str.contains(ql, na=False)\n        return df[mask]\n    else:\n        if field in df.columns:\n            return df[df[field].astype(str).str.lower().str.contains(ql, na=False)]\n        return pd.DataFrame()\n \nfeat_cols = [c for c in [\n    "danceability","energy","loudness","speechiness","acousticness",\n    "instrumentalness","liveness","valence","tempo","duration_ms","track_popularity"\n] if c in df.columns]\nnn = None; X = None\nif len(feat_cols) >= 2:\n    scaler = StandardScaler()\n    X = scaler.fit_transform(df[feat_cols].fillna(df[feat_cols].median()).values)\n    nn = NearestNeighbors(n_neighbors=11, metric="euclidean")\n    nn.fit(X)\n\nif st.button("Search"):\n    res = search_rows(query, search_field)\n    if res.empty:\n        st.error(f"Not Found — no matches for \'{query}\' in \'{search_field}\'.")\n    else:\n        show = res.head(200)\n        for idx, row in show.iterrows():\n        \n            st.markdown(f"""\n            <div class="song-card">\n              <div><b>{row.get(\'track_name\',\'\')}</b> — {row.get(\'track_artist\',\'\')}</div>\n              <div>Album: {row.get(\'track_album_name\',\'\')}</div>\n              <div>Playlist: {row.get(\'playlist_name\',\'\')} | Genre: {row.get(\'playlist_genre\',\'\')} | Subgenre: {row.get(\'playlist_subgenre\',\'\')}</div>\n            </div>\n            """, unsafe_allow_html=True)\n\n            yt_q = urllib.parse.quote_plus(f"{row.get(\'track_name\',\'\')} {row.get(\'track_artist\',\'\')}")\n            yt_link = f"https://www.youtube.com/results?search_query={yt_q}"\n            st.markdown(f"🔗 **YouTube (Play/Search first):** [{row.get(\'track_name\',\'\')} on YouTube]({yt_link})")\n\n            tid = str(row.get("track_id","")).strip()\n            if tid and tid.lower() != "unknown":\n                sp_link = f"https://open.spotify.com/track/{tid}"\n            else:\n                sp_q = urllib.parse.quote_plus(f"{row.get(\'track_name\',\'\')} {row.get(\'track_artist\',\'\')}")\n                sp_link = f"https://open.spotify.com/search/{sp_q}"\n            st.markdown(f"🎧 **Spotify (second):** [{row.get(\'track_name\',\'\')} on Spotify]({sp_link})")\n \n            if nn is not None and X is not None:\n                try:\n                    distances, indices = nn.kneighbors(X[idx:idx+1], n_neighbors=11)\n                    neighbors = indices.flatten().tolist()\n                    if neighbors and neighbors[0] == idx:\n                        neighbors = neighbors[1:11]\n                    else:\n                        neighbors = neighbors[:10]\n                    recs = df.iloc[neighbors]\n                    with st.expander("Show 10 similar songs"):\n                        for _, r in recs.iterrows():\n                            st.write(f"- {r.get(\'track_name\',\'\')} — {r.get(\'track_artist\',\'\')} ({r.get(\'playlist_genre\',\'\')})")\n                except Exception:\n                    pass\n\nst.markdown("---")\nst.caption("YouTube link appears first; Spotify second. Use \'Any\' to search across all fields.")\n')


# In[15]:


get_ipython().system('pip install pyngrok streamlit -q')

import os
import subprocess
import time
from pyngrok import ngrok

NGROK_TOKEN = "31KPNKThRlF8Ve4fhErGw15OBhn_6Dp2eZfKvqx94zp1tTRjZ"  # apna token paste karo
os.environ["NGROK_AUTHTOKEN"] = NGROK_TOKEN
ngrok.set_auth_token(NGROK_TOKEN)

try:
    subprocess.run(["fuser", "-k", "8501/tcp"])
except:
    pass

subprocess.Popen(["streamlit", "run", "spotify_app.py", "--server.port", "8501", "--server.headless", "true"])

public_url = ngrok.connect(8501)
print("🌍 Public URL:", public_url)


# In[ ]:




