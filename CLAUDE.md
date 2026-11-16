# Project progress file (read this first in a new chat)

Project: AI-Based Predictive Maintenance (AI4I 2020, scikit-learn + XGBoost + Streamlit)
Repo: github.com/purnakshibansal/AI-Based-Predictive-Maintenance (private, branch main)
Workspace: GitHub Codespace, /workspaces/AI-Based-Predictive-Maintenance, Python 3.14.2

## Status
- Done and pushed: phases 0-10 (setup, data, packages, structure, preprocessing, EDA,
  training, evaluation, final model + predict.py, all seven dashboard pages).
- Done: README.md, this file.
- Remaining: final dashboard review, deployment on Streamlit Community Cloud.
- Tests: 64 pytest tests passing, ruff clean.

## Key decisions
- Model inputs: Type + 5 sensors only. Never UDI, Product ID, Machine failure or TWF/HDF/PWF/OSF/RNF.
- Outliers are never deleted (torque outliers fail ~90% of the time).
- Final model chosen by cross-validated PR-AUC BEFORE using test rows: XGBoost.
- Threshold chosen on out-of-fold TRAINING predictions, maximising F2 (recall counts double).
- Final model trained on the 8,000 training rows only, so test numbers stay honest.
- Test results: 54/68 failures found, 14 missed, 34 false alarms, PR-AUC 0.827.
- Scores are "risk indicators", never "probabilities". Data is synthetic; say so in the UI.
- Raw CSV is committed (CC BY 4.0, attribution in data/README.md).
- models/candidates/ is git-ignored (large, re-creatable); models/best_model.joblib is committed.

## Gotchas
- New terminals need: source .venv/bin/activate (prompt must start with (.venv)).
- CSV has a BOM: load with encoding="utf-8-sig" (data_loader does this).
- pandas 3.x; XGBoost needs the xgboost-cpu package (imports as xgboost).
- Preprocessor outputs a NumPy array on purpose (XGBoost rejects "[" in column names).
- Streamlit 1.67: use width="stretch", not use_container_width.
- Dashboard pages live in views/, wired together in app.py with st.navigation.
- Write files with cat > file <<'EOF' blocks; never paste code straight into the terminal.
- Commit by naming files (no git add .); check no .venv, caches or models/candidates are staged.

## Commands
- Tests/lint: python -m pytest -q ; ruff check .
- Run app: streamlit run app.py
- Retrain: python -m src.train_model ; python -m src.evaluate_model ; python -m src.finalize_model

## Deployment notes
- Streamlit Community Cloud picks Python in Advanced settings (runtime.txt is ignored).
- Requirements are pinned for Python 3.14; choose 3.14 if offered, otherwise check the build log.
