import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="CineMatch — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom CSS — navy / purple / teal cinematic theme
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #0a0e1a 0%, #131025 45%, #1b1535 100%);
        color: #E8E8F0;
    }
    h1, h2, h3 { color: #F5F5FF !important; font-weight: 700 !important; }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7C5CFF, #22D3C5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub { color: #9C9CB8; font-size: 1rem; margin-top: 0.2rem; }
    section[data-testid="stSidebar"] { background: #0d0b1a; border-right: 1px solid #2a2450; }
    section[data-testid="stSidebar"] * { color: #E8E8F0 !important; }
    .rec-card {
        background: linear-gradient(135deg, #171331 0%, #1e1840 100%);
        border: 1px solid #322a5e;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .rec-card:hover { transform: translateY(-2px); border-color: #7C5CFF; }
    .rec-title { font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px; }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge.confidence { background: rgba(34, 211, 197, 0.15); color: #22D3C5; border: 1px solid #22D3C5; }
    .badge.lift { background: rgba(124, 92, 255, 0.15); color: #A78BFA; border: 1px solid #7C5CFF; }
    .badge.support { background: rgba(245, 158, 11, 0.12); color: #F5A623; border: 1px solid #F5A623; }
    div[data-testid="stMetric"] { background: #171331; border: 1px solid #2a2450; border-radius: 12px; padding: 12px 16px; }
    div[data-baseweb="select"] > div { background-color: #171331 !important; border-color: #322a5e !important; color: #E8E8F0 !important; }
    .stButton>button {
        background: linear-gradient(90deg, #7C5CFF, #22D3C5);
        color: #0a0e1a; font-weight: 700; border: none; border-radius: 8px;
    }
    .stButton>button:hover { opacity: 0.9; color: #0a0e1a; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="hero-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-powered movie recommendations — FP-Growth association rule mining engine, running live</div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------
# Sidebar — data source + engine parameters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Data")
    movies_file = st.file_uploader("movies.csv", type="csv")
    ratings_file = st.file_uploader("ratings.csv (or .zip)", type=["csv", "zip"])
    use_defaults = st.checkbox(
        "Use local movies.csv / ratings.csv instead",
        value=not (movies_file or ratings_file),
        help="Looks for movies.csv and ratings.csv in the same folder as app.py",
    )

    st.markdown("---")
    st.markdown("### 🎛️ Engine parameters")
    min_rating = st.slider("Minimum rating counted as 'liked'", 1.0, 5.0, 4.0, 0.5)
    min_users = st.slider("Min users who liked a movie (noise filter)", 5, 100, 40, 5)
    min_support = st.slider("Min support (FP-Growth)", 0.01, 0.2, 0.04, 0.01)
    min_confidence = st.slider("Min confidence (rules)", 0.1, 0.9, 0.5, 0.05)

    st.markdown("---")
    st.markdown("### 🔎 Recommend")
    top_n = st.slider("Number of recommendations", 3, 20, 8)
    sort_by = st.radio("Rank by", ["confidence", "lift", "support"], index=0)

    run_btn = st.button("🚀 Build / rebuild engine", use_container_width=True)


# ---------------------------------------------------------
# Recommendation engine (cached — only reruns when inputs change)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw(movies_bytes, ratings_bytes, use_local):
    if use_local:
        movies = pd.read_csv("movies.csv")
        ratings = pd.read_csv("ratings.csv")
    else:
        movies = pd.read_csv(movies_bytes)
        ratings = pd.read_csv(ratings_bytes)
    return movies, ratings


@st.cache_data(show_spinner=False)
def build_rules(movies, ratings, min_rating, min_users, min_support, min_confidence):
    # merge + clean, same pipeline as the notebook
    df = pd.merge(ratings, movies, on="movieId", how="inner")
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    df = df[df["rating"] >= min_rating]

    # build "transactions": each user's list of liked movies
    transactions = df.groupby("userId")["title"].apply(list).reset_index()

    # one-hot encode
    te = TransactionEncoder()
    te_array = te.fit(transactions["title"]).transform(transactions["title"])
    onehot = pd.DataFrame(te_array, columns=te.columns_)

    # filter out rarely-liked movies (reduces noise/sparsity)
    movie_count = onehot.sum(axis=0)
    selected_movies = movie_count[movie_count >= min_users].index
    onehot_small = onehot[selected_movies]

    # frequent itemsets via FP-Growth
    frequent_itemsets = fpgrowth(onehot_small, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    # association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    return rules


# ---------------------------------------------------------
# Load data + run engine
# ---------------------------------------------------------
if use_defaults:
    try:
        movies_df, ratings_df = load_raw(None, None, True)
    except FileNotFoundError:
        st.error(
            "⚠️ Couldn't find `movies.csv` / `ratings.csv` in this folder. "
            "Either place both files next to `app.py`, or upload them via the sidebar."
        )
        st.stop()
elif movies_file and ratings_file:
    movies_df, ratings_df = load_raw(movies_file, ratings_file, False)
else:
    st.info("👈 Upload `movies.csv` and `ratings.csv` in the sidebar (or check 'Use local files') to build the recommendation engine.")
    st.stop()

with st.spinner("Mining frequent itemsets and building association rules..."):
    rules = build_rules(movies_df, ratings_df, min_rating, min_users, min_support, min_confidence)

if rules.empty:
    st.warning(
        "No rules generated with these settings. Try lowering **min support** or **min confidence**, "
        "or lowering the **min users** noise filter in the sidebar."
    )
    st.stop()

all_movies = sorted(rules["antecedents"].unique())

# ---------------------------------------------------------
# Movie picker
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    search = st.text_input("🔍 Search a movie", "")
    filtered_movies = [m for m in all_movies if search.lower() in m.lower()] or all_movies
    selected_movie = st.selectbox("Pick a movie you loved", filtered_movies)

    st.subheader(f"Because you liked **{selected_movie}** →")

    matches = (
        rules[rules["antecedents"] == selected_movie]
        .sort_values(by=sort_by, ascending=False)
        .head(top_n)
    )

    if matches.empty:
        st.warning("No association rules found for this movie. Try another title, or loosen the engine parameters in the sidebar.")
    else:
        for _, row in matches.iterrows():
            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-title">🎞️ {row['consequents']}</div>
                    <span class="badge confidence">Confidence {row['confidence']*100:.1f}%</span>
                    <span class="badge lift">Lift {row['lift']:.2f}</span>
                    <span class="badge support">Support {row['support']*100:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

with col2:
    st.subheader("📊 Engine stats")
    st.metric("Total rules generated", f"{len(rules):,}")
    st.metric("Unique movies (antecedents)", f"{len(all_movies):,}")
    st.metric("Users in dataset", f"{ratings_df['userId'].nunique():,}")

    st.subheader("📈 Top overall rules")
    top_overall = rules.sort_values(by="lift", ascending=False).head(10)[
        ["antecedents", "consequents", "lift"]
    ]
    st.dataframe(top_overall, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Explore all rules + download
# ---------------------------------------------------------
with st.expander("🔍 Explore all association rules"):
    st.dataframe(
        rules.sort_values(by="confidence", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "⬇️ Download rules as CSV",
        rules.to_csv(index=False).encode("utf-8"),
        file_name="movie_association_rules.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Built with Streamlit · FP-Growth association rule mining engine runs live in this app")
