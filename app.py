import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")
CSV_PATH = "IMDb_All_Genres_Movies_cleaned.csv"
RANDOM_STATE = 42
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"
]

EMOTION_TO_GENRES = {
    "Happy": ["Comedy", "Family", "Musical"],
    "Sad": ["Drama", "Romance"],
    "Angry": ["Action", "Crime", "Thriller"],
    "Calm": ["Documentary", "Drama", "Romance"],
    "Excited": ["Action", "Adventure"],
    "Romantic": ["Romance", "Drama"],
    "Scared": ["Horror", "Thriller"],
    "Nostalgic": ["Drama", "History"],
    "Curious": ["Mystery", "Sci-Fi", "Documentary"],
    "Bored": ["Comedy", "Adventure", "Action"]
}

def normalize_string(s):
    return re.sub(r'\s+',' ', str(s).strip())

def split_genres_field(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [p.strip().title() for p in re.split(r'[,/;|]', str(s)) if p.strip()]

def safe_year(y):
    try:
        return int(float(y))
    except:
        return 0

def load_csv(path):
    if not os.path.exists(path):
        st.error(f"CSV file not found: {path}")
        st.stop()
    return pd.read_csv(path)

def clean_df(df):
    df = df.copy()
    for c in ["Movie_Title", "Actors", "Director", "main_genre", "side_genre", "Rating", "Runtime(Mins)", "Year"]:
        if c not in df.columns:
            df[c] = ""
    df["Movie_Title"] = df["Movie_Title"].astype(str).apply(normalize_string)
    df["Actors"] = df["Actors"].astype(str).fillna("")
    df["Director"] = df["Director"].astype(str).fillna("")
    df["main_genre"] = df["main_genre"].astype(str).fillna("")
    df["side_genre"] = df["side_genre"].astype(str).fillna("")
    df["main_tokens"] = df["main_genre"].apply(split_genres_field)
    df["side_tokens"] = df["side_genre"].apply(split_genres_field)
    df["all_genres"] = df.apply(lambda r: sorted(set(r["main_tokens"] + r["side_tokens"])), axis=1)
    df["primary_genre"] = df["main_tokens"].apply(lambda x: x[0] if x else "Unknown")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)
    df["Runtime(Mins)"] = pd.to_numeric(df["Runtime(Mins)"], errors="coerce").fillna(0)
    df["Year"] = df["Year"].apply(safe_year)
    df["combined_text"] = (
        df["Movie_Title"] + " | " + df["Actors"] + " | " +
        df["Director"] + " | " + df["main_genre"] + " | " + df["side_genre"]
    )
    df["title_key"] = df["Movie_Title"].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True).str.strip()
    df = df.sort_values(["Rating", "Runtime(Mins)"], ascending=[False, True])
    df = df.drop_duplicates(subset=["title_key"]).reset_index(drop=True)
    return df

@st.cache_data
def build_tfidf_svd(corpus, max_features=5000):
    vec = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vec.fit_transform(corpus)
    n_components = min(50, max(5, len(corpus)//4))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    Xr = svd.fit_transform(X)
    return vec, svd, Xr

def genre_color_map(genres):
    unique = sorted(set(genres))
    return {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(unique)}

def badge_html(text, color):
    return f'<span style="background:{color}; color:#fff; padding:3px 8px; border-radius:10px; font-size:12px;">{text}</span>'

def build_final_projection(features, method="PCA", n_components=10, labels=None):
    if method == "PCA":
        model = PCA(n_components=min(n_components, features.shape[1]), random_state=RANDOM_STATE)
        return model.fit_transform(features), model
    if labels is None or len(np.unique(labels)) < 2:
        model = PCA(n_components=min(n_components, features.shape[1]), random_state=RANDOM_STATE)
        return model.fit_transform(features), model
    try:
        lda = LinearDiscriminantAnalysis(n_components=min(n_components, len(np.unique(labels)) - 1))
        return lda.fit_transform(features, labels), lda
    except:
        model = PCA(n_components=min(n_components, features.shape[1]), random_state=RANDOM_STATE)
        return model.fit_transform(features), model

def recommend_by_index(index, final_space, df_filtered, top_n=8):
    sims = cosine_similarity(final_space[index:index+1], final_space).flatten()
    order = np.argsort(-sims)
    recs = []
    seen = set()
    for i in order:
        if i == index:
            continue
        title = df_filtered.iloc[i]["Movie_Title"]
        if title in seen:
            continue
        seen.add(title)
        recs.append((i, title, float(sims[i])))
        if len(recs) >= top_n:
            break
    return recs

# --- DATA LOADING ---
raw = load_csv(CSV_PATH)
df = clean_df(raw)

# --- SESSION STATE: For expanded recommendations ---
if "expanded_recs" not in st.session_state:
    st.session_state.expanded_recs = []

# --- SIDEBAR: Controls ---
with st.sidebar:
    st.header("Controls")
    emotion = st.selectbox("What are you feeling?", options=[""] + list(EMOTION_TO_GENRES.keys()), key="w_emotion")
    search_type = st.radio("Search by", ["All", "Actor", "Director", "Title"], index=0, key="w_search_type")
    titles_list = sorted(df["Movie_Title"].astype(str).unique())
    actors_list = sorted({a.strip() for x in df["Actors"] for a in str(x).split(",") if a.strip()})
    directors_list = sorted({d.strip() for x in df["Director"] for d in str(x).split(",") if d.strip()})
    if search_type == "Actor":
        search_items = actors_list
    elif search_type == "Director":
        search_items = directors_list
    elif search_type == "Title":
        search_items = titles_list
    else:
        search_items = sorted(titles_list + actors_list + directors_list)
    search_q = st.selectbox("Search title / actor / director", options=[""] + search_items, key="w_search_q")
    rec_mode = st.radio("Recommendation mode", ["Similarity (PCA/LDA)", "Same actor/director"], index=0, key="w_rec_mode")
    final_method = st.radio("Final projection", ["PCA", "LDA"], index=0, key="w_final_method")
    tfidf_max = st.number_input("TF-IDF max features", 200, 20000, 5000, 100, key="w_tfidf_max")
    final_comp = st.number_input("Final components", 2, 50, 10, 1, key="w_final_comp")
    min_rating = st.slider("Min rating", 0.0, 10.0, 0.0, 0.1, key="w_min_rating")
    year_min, year_max = st.slider(
        "Year range",
        int(df["Year"].min()),
        int(df["Year"].max()),
        (int(df["Year"].min()), int(df["Year"].max())),
        key="w_year_range",
    )
    chosen_genre = st.selectbox(
        "Filter by genre (optional)", options=[""] + sorted({g for gs in df["all_genres"] for g in gs}), key="w_chosen_genre"
    )
    rec_count = st.number_input("Number of recommendations", 1, 50, 8, 1, key="w_rec_count")
    if st.button("Run genre-consistency test (Top-5)"):
        st.session_state.do_accuracy = True

# --- FILTER STATE ---
rec_mode = st.session_state.get("w_rec_mode", "Similarity (PCA/LDA)")
emotion = st.session_state.get("w_emotion", "")
search_q = st.session_state.get("w_search_q", "").strip()
search_type = st.session_state.get("w_search_type", "All")
tfidf_max = st.session_state.get("w_tfidf_max", 5000)
final_method = st.session_state.get("w_final_method", "PCA")
final_comp = st.session_state.get("w_final_comp", 10)
min_rating = st.session_state.get("w_min_rating", 0.0)
year_min, year_max = st.session_state.get("w_year_range", (int(df["Year"].min()), int(df["Year"].max())))
chosen_genre = st.session_state.get("w_chosen_genre", "")
rec_count = int(st.session_state.get("w_rec_count", 8))

# --- APPLY FILTERS ---
mask = (df["Rating"] >= min_rating) & df["Year"].between(year_min, year_max)
if emotion:
    allowed = EMOTION_TO_GENRES.get(emotion, [])
    mask &= df["all_genres"].apply(lambda gs: any(g in gs for g in allowed))
if chosen_genre:
    mask &= df["all_genres"].apply(lambda gs: chosen_genre in gs)
if search_q:
    sq = search_q.lower()
    if search_type == "Actor":
        mask &= df["Actors"].apply(lambda x: any(sq == a.strip().lower() for a in str(x).split(",") if a.strip()))
    elif search_type == "Director":
        mask &= df["Director"].apply(lambda x: any(sq == d.strip().lower() for d in str(x).split(",") if d.strip()))
    elif search_type == "Title":
        mask &= df["Movie_Title"].apply(lambda t: sq in str(t).lower())
    else:
        mask &= df.apply(lambda r: (sq in str(r["Movie_Title"]).lower()) or (sq in str(r["Actors"]).lower()) or (sq in str(r["Director"]).lower()), axis=1)

filtered = df[mask].reset_index(drop=True)
st.write(f"Filtered movies: {len(filtered)}")
if len(filtered) == 0:
    st.warning("No movies found after filtering.")
    st.stop()

tfidf_vec, svd, _ = build_tfidf_svd(df["combined_text"].tolist(), max_features=tfidf_max)
Xf = tfidf_vec.transform(filtered["combined_text"].tolist())
text_vecs = svd.transform(Xf)
num = np.vstack([filtered["Rating"], filtered["Runtime(Mins)"], filtered["Year"]]).T
scaler = StandardScaler().fit(num)
num_scaled = scaler.transform(num)
features = np.hstack([text_vecs, num_scaled])
labels = filtered["primary_genre"].values if final_method == "LDA" else None
final_space, final_model = build_final_projection(features, final_method, n_components=min(final_comp, features.shape[1]), labels=labels)

# --- SCATTER PLOT ---
st.subheader("2D Projection (Comp1 vs Comp2)")
genres = filtered["primary_genre"].values
cmap = genre_color_map(genres)
colors = [cmap.get(g, "#777777") for g in genres]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(final_space[:, 0], final_space[:, 1], c=colors, s=40, edgecolor='k', linewidth=0.3)
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_title("Movies 2D Projection")
handles = [Line2D([0], [0], marker='o', color='w', label=g, markerfacecolor=cmap[g], markersize=8) for g in sorted(set(genres))[:16]]
ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Primary Genre")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

# --- RECOMMENDATION FUNCTIONS ---
def show_recs(recs, df_local=filtered, expand_level=1):
    for rank, (i, title, score) in enumerate(recs, start=1):
        r = df_local.iloc[i]
        badge = badge_html(r['primary_genre'], cmap.get(r['primary_genre'], "#444444"))
        st.markdown(
            f"{rank}. {badge} <b>{title}</b> — ⭐ {r['Rating']} | {r['Year']} — Sim: {score:.3f}",
            unsafe_allow_html=True
        )
        st.write(f"Director: {r['Director']} | Runtime: {int(r['Runtime(Mins)'])} mins")

        # Expansion for this recommendation
        expand_key = f"expand_{title}_{rank}_{expand_level}"
        if st.button(f"Show recommendations for '{title}'", key=expand_key):
            if (title, expand_level) not in st.session_state.expanded_recs:
                st.session_state.expanded_recs.append((title, expand_level))
                st.experimental_rerun()

        # If expanded, show its recommendations just below
        if (title, expand_level) in st.session_state.expanded_recs:
            movie_index = df_local.index[df_local["Movie_Title"] == title][0]
            recs_nested = recommend_by_index(movie_index, final_space, df_local, top_n=5)
            st.markdown(f"**Recommendations for _{title}_:**")
            for n_rank, (ni, n_title, n_score) in enumerate(recs_nested, start=1):
                n_r = df_local.iloc[ni]
                n_badge = badge_html(n_r['primary_genre'], cmap.get(n_r['primary_genre'], "#444444"))
                st.markdown(f"&nbsp;&nbsp;{n_rank}. {n_badge} <b>{n_title}</b> — ⭐ {n_r['Rating']} | {n_r['Year']} — Sim: {n_score:.3f}", unsafe_allow_html=True)
                st.write(f"&nbsp;&nbsp;Director: {n_r['Director']} | Runtime: {int(n_r['Runtime(Mins)'])} mins")
            st.markdown("---")
        st.markdown("---")

mode = st.radio("Query source", ("Existing movie", "Custom entry"), index=0, key="w_mode")

if mode == "Existing movie":
    sel = st.selectbox("Pick a movie", options=[""] + filtered["Movie_Title"].tolist(), key="w_pick_movie")
    if sel:
        pos = int(filtered.index[filtered["Movie_Title"] == sel][0])
        if rec_mode == "Same actor/director":
            base = filtered.iloc[pos]
            base_actors = [a.strip().lower() for a in str(base["Actors"]).split(",") if a.strip()]
            base_directors = [d.strip().lower() for d in str(base["Director"]).split(",") if d.strip()]
            def shares_person(row):
                s = str(row["Actors"]).lower() + " " + str(row["Director"]).lower()
                return any(a in s for a in base_actors) or any(d in s for d in base_directors)
            mask_shared = filtered.apply(shares_person, axis=1)
            mask_shared &= (filtered["Movie_Title"] != sel)
            idxs = list(filtered[mask_shared].index)[:rec_count]
            st.subheader(f"Movies sharing actor/director with {sel}")
            if not idxs:
                st.info("No shared actors/directors found in current filtered set.")
            else:
                show_recs([(i, filtered.iloc[i]["Movie_Title"], 1.0) for i in idxs], filtered)
        else:
            recs = recommend_by_index(pos, final_space, filtered, top_n=rec_count)
            st.subheader(f"Top {len(recs)} similar to {sel}")
            show_recs(recs, filtered)
else:
    st.write("Provide a custom movie entry:")
    with st.form("custom"):
        c_title = st.text_input("Title", "My Movie")
        c_actors = st.text_input("Actors (comma separated)", "")
        c_director = st.text_input("Director", "")
        c_main = st.text_input("Main genre", "")
        c_side = st.text_input("Side genre", "")
        c_rating = st.number_input("Rating", 0.0, 10.0, 6.0, step=0.1)
        c_runtime = st.number_input("Runtime", 0, 600, 120, step=1)
        c_year = st.number_input("Year", 1800, 2100, 2020, step=1)
        submitted = st.form_submit_button("Find Recommendations")
    if submitted:
        combined = f"{c_title} | {c_actors} | {c_director} | {c_main} | {c_side}"
        q_text = tfidf_vec.transform([combined])
        q_text_red = svd.transform(q_text)
        q_num = np.array([[c_rating, c_runtime, c_year]])
        q_num_scaled = scaler.transform(q_num)
        q_feat = np.hstack([q_text_red, q_num_scaled])
        try:
            q_proj = final_model.transform(q_feat)
            sims = cosine_similarity(q_proj, final_space).flatten()
        except:
            sims = cosine_similarity(q_feat, features).flatten()
        idxs = np.argsort(-sims)[:rec_count]
        recs = [(int(i), filtered.iloc[i]["Movie_Title"], float(sims[i])) for i in idxs]
        st.write(f"Top {len(recs)} recommendations for your custom movie:")
        show_recs(recs, filtered)

# --- GENRE CONSISTENCY TEST ---
if st.session_state.get("do_accuracy", False):
    st.session_state.do_accuracy = False
    with st.spinner("Computing genre-consistency (Top-5)..."):
        genre_acc, avg_sim = evaluate_genre_consistency(final_space, filtered, top_n=5)
        st.success(f"🎯 Genre-consistency (Top-5): {genre_acc*100:.2f}%")
        st.info(f"Average cosine similarity (Top-5): {avg_sim:.4f}")
