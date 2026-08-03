"""
Movie Recommender — Streamlit App
FP-Growth / Apriori association rules over MovieLens data.
Reads movie_association_rules.csv (columns: antecedents, consequents,
support, confidence, lift — antecedents/consequents stored as
frozenset repr strings, e.g. "frozenset({'Movie A', 'Movie B'})").
"""

import streamlit as st
import pandas as pd
import ast
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ---------------- theme (dark navy / purple / teal) ----------------
st.markdown("""
<style>
:root {
    --bg: #0b0f1a;
    --panel: #131a2b;
    --border: #232b40;
    --purple: #8b5cf6;
    --teal: #2dd4bf;
    --text: #e5e7eb;
    --muted: #8b93a7;
}
.stApp {
    background: radial-gradient(circle at top, #101627 0%, var(--bg) 60%);
    color: var(--text);
}
h1, h2, h3 { color: var(--text) !important; }
.title-grad {
    font-size: 30px;
    font-weight: 700;
    background: linear-gradient(90deg, var(--purple), var(--teal));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0px;
}
.subtitle { color: var(--muted); font-size: 14px; margin-bottom: 24px; }
div[data-baseweb="select"] > div {
    background-color: #0e1424 !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(139,92,246,0.18) !important;
    border: 1px solid var(--purple) !important;
    color: var(--purple) !important;
}
.stButton > button {
    background: linear-gradient(90deg, var(--purple), var(--teal));
    color: #0b0f1a;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
}
.stButton > button:hover { opacity: 0.9; }
.rec-card {
    background: rgba(45,212,191,0.08);
    border: 1px solid var(--border);
    border-left: 3px solid var(--teal);
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 10px;
}
.rec-card .rec-name { font-size: 15px; font-weight: 600; color: var(--text); }
.rec-card .rec-meta { font-size: 12px; color: var(--muted); margin-top: 2px; }
.empty-box {
    background: var(--panel);
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    color: var(--muted);
}
</style>
""", unsafe_allow_html=True)


# ---------------- load + parse rules ----------------
def parse_frozenset_str(s: str) -> frozenset:
    """Parses "frozenset({'A', 'B'})" strings back into real frozensets."""
    match = re.search(r"frozenset\((.*)\)", s)
    inner = match.group(1) if match else s
    return frozenset(ast.literal_eval(inner))


@st.cache_resource
def load_data():
    df = pd.read_csv(BASE_DIR / "movie_association_rules.csv")
    df["antecedents"] = df["antecedents"].apply(parse_frozenset_str)
    df["consequents"] = df["consequents"].apply(parse_frozenset_str)

    movies = set()
    for col in ("antecedents", "consequents"):
        for fs in df[col]:
            movies.update(fs)

    return df, sorted(movies)


rules, movies = load_data()


# ---------------- recommend logic ----------------
def get_recommendations(watched: list[str], top_n: int = 5) -> pd.DataFrame:
    watched_set = frozenset(watched)

    matched = rules[rules["antecedents"].apply(lambda a: a.issubset(watched_set))].copy()
    if matched.empty:
        return pd.DataFrame(columns=["movie", "confidence", "lift"])

    matched = matched.sort_values(by=["lift", "confidence"], ascending=False)

    seen = set(watched_set)
    rows = []
    for _, r in matched.iterrows():
        for m in r["consequents"]:
            if m not in seen:
                seen.add(m)
                rows.append({"movie": m, "confidence": r["confidence"], "lift": r["lift"]})
            if len(rows) >= top_n:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


# ---------------- UI ----------------
st.markdown('<div class="title-grad">Movie Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Association rule mining on MovieLens · pick movies you\'ve watched to get recommendations</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 1])
with col1:
    watched = st.multiselect("Movies you've watched", options=movies, placeholder="Select movies...")
with col2:
    top_n = st.number_input("Top N", min_value=1, max_value=10, value=5, step=1)

get_recs = st.button("Get Recommendations", disabled=len(watched) == 0, width='stretch')

st.markdown("---")

if get_recs:
    recs_df = get_recommendations(watched, top_n)
    if recs_df.empty:
        st.markdown(
            '<div class="empty-box">No matching rules for this exact combination. '
            'This ruleset only has a handful of high-confidence rules — try selecting the '
            'movies from one of the rule sets shown in "All rules" below.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**Because you watched:** {', '.join(watched)}")
        for _, row in recs_df.iterrows():
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-name">{row['movie']}</div>
                <div class="rec-meta">confidence: {row['confidence']:.2f} · lift: {row['lift']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
elif not watched:
    st.markdown(
        '<div class="empty-box">Select one or more movies above to see recommendations.</div>',
        unsafe_allow_html=True,
    )

with st.expander("Browse all movies in this ruleset"):
    st.write(f"{len(movies)} movies")
    st.dataframe(pd.DataFrame({"movie": movies}), width='stretch', hide_index=True)

with st.expander("All rules"):
    display_rules = rules.copy()
    display_rules["antecedents"] = display_rules["antecedents"].apply(lambda x: ", ".join(x))
    display_rules["consequents"] = display_rules["consequents"].apply(lambda x: ", ".join(x))
    st.dataframe(
        display_rules[["antecedents", "consequents", "support", "confidence", "lift"]]
        .sort_values("lift", ascending=False),
        width='stretch',
        hide_index=True,
    )