import streamlit as st
import pandas as pd

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

    /* Headings */
    h1, h2, h3 {
        color: #F5F5FF !important;
        font-weight: 700 !important;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7C5CFF, #22D3C5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #9C9CB8;
        font-size: 1rem;
        margin-top: 0.2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d0b1a;
        border-right: 1px solid #2a2450;
    }
    section[data-testid="stSidebar"] * {
        color: #E8E8F0 !important;
    }

    /* Recommendation card */
    .rec-card {
        background: linear-gradient(135deg, #171331 0%, #1e1840 100%);
        border: 1px solid #322a5e;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        border-color: #7C5CFF;
    }
    .rec-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge.confidence {
        background: rgba(34, 211, 197, 0.15);
        color: #22D3C5;
        border: 1px solid #22D3C5;
    }
    .badge.lift {
        background: rgba(124, 92, 255, 0.15);
        color: #A78BFA;
        border: 1px solid #7C5CFF;
    }
    .badge.support {
        background: rgba(245, 158, 11, 0.12);
        color: #F5A623;
        border: 1px solid #F5A623;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #171331;
        border: 1px solid #2a2450;
        border-radius: 12px;
        padding: 12px 16px;
    }

    /* Selectbox / inputs */
    div[data-baseweb="select"] > div {
        background-color: #171331 !important;
        border-color: #322a5e !important;
        color: #E8E8F0 !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #7C5CFF, #22D3C5);
        color: #0a0e1a;
        font-weight: 700;
        border: none;
        border-radius: 8px;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: #0a0e1a;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data
def load_rules(path="movie_association_rules.csv"):
    df = pd.read_csv(path)
    df["confidence"] = df["confidence"].astype(float)
    df["lift"] = df["lift"].astype(float)
    df["support"] = df["support"].astype(float)
    return df


try:
    rules = load_rules()
except FileNotFoundError:
    st.error(
        "⚠️ `movie_association_rules.csv` not found. Export it from your notebook "
        "(`rules.to_csv('movie_association_rules.csv', index=False)`) and place it "
        "in the same folder as `app.py`."
    )
    st.stop()

all_movies = sorted(rules["antecedents"].unique())

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="hero-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-powered movie recommendations using Association Rule Mining (FP-Growth on MovieLens)</div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    search = st.text_input("Search a movie", "")
    filtered_movies = [m for m in all_movies if search.lower() in m.lower()] or all_movies
    selected_movie = st.selectbox("Pick a movie you loved", filtered_movies)
    top_n = st.slider("Number of recommendations", 3, 20, 8)
    sort_by = st.radio("Rank recommendations by", ["confidence", "lift", "support"], index=0)

    st.markdown("---")
    st.markdown("### 📊 Dataset stats")
    st.metric("Total rules", f"{len(rules):,}")
    st.metric("Unique movies (antecedents)", f"{len(all_movies):,}")

# ---------------------------------------------------------
# Main content
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Because you liked **{selected_movie}** →")

    matches = (
        rules[rules["antecedents"] == selected_movie]
        .sort_values(by=sort_by, ascending=False)
        .head(top_n)
    )

    if matches.empty:
        st.warning("No association rules found for this movie. Try another title or lower the min_support/min_threshold when generating rules.")
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
    st.subheader("📈 Top overall rules")
    top_overall = rules.sort_values(by="lift", ascending=False).head(10)[
        ["antecedents", "consequents", "lift"]
    ]
    st.dataframe(top_overall, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Explore all rules
# ---------------------------------------------------------
with st.expander("🔍 Explore all association rules"):
    st.dataframe(
        rules.sort_values(by="confidence", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")
st.caption("Built with Streamlit · Recommendations generated via FP-Growth association rule mining")