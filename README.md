# Movie Recommender — Streamlit

Association-rule-based movie recommender from your MovieLens FP-Growth work.

## Files
```
movie_recommender_streamlit/
├── app.py
├── requirements.txt
└── movie_association_rules.csv
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo — `app.py`, `requirements.txt`,
   `movie_association_rules.csv` at the same level.
2. share.streamlit.io → **New app** → connect repo → Main file path = `app.py`.
3. Deploy.

Same `requirements.txt` filename gotcha as your other projects — exact
filename, no spaces.

## What it does
- Parses `antecedents`/`consequents` from the CSV. Your export saved them
  as Python `frozenset({...})` repr **strings**, not real objects — CSV
  can't hold Python objects. `parse_frozenset_str()` regexes out the
  `frozenset(...)` wrapper and uses `ast.literal_eval` to turn it back into
  a real `frozenset`. If you ever export straight to `.pkl` with `joblib`
  instead, you can skip this parsing step entirely — worth doing for future
  projects, it's cleaner and faster.
- Multi-select "movies you've watched" → subset-matches against
  `antecedents`, ranks by lift then confidence, same logic as your market
  basket project.
- Two expanders: browse all movies in the ruleset, and view all rules
  sorted by lift.

## Heads-up on your dataset
This CSV only has **6 rules** covering **10 movies**, and every rule has a
3–4 movie antecedent. That means most single or double movie picks will
return "no matching rules" — you'll only get recommendations when the
selected movies happen to match one of those larger antecedent sets. Check
the "All rules" expander in the app to see exactly which combinations
trigger a result. If this is a full export (not a trimmed sample), you
might want to lower `min_support` in your notebook and regenerate with more
rules before this feels good on LinkedIn — a demo where most searches
return empty isn't a great first impression for recruiters.