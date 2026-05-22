import json

with open('testTask/notebooks/pipeline_with_clustering.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 23 - simplified grid search
nb['cells'][23] = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'from sklearn.experimental import enable_halving_search_cv\n',
        'from sklearn.model_selection import HalvingGridSearchCV\n',
        '\n',
        '# --- Базовая модель ---\n',
        'baseline = LogisticRegression(max_iter=1000, random_state=42)\n',
        'baseline.fit(x_train_tfidf, y_train)\n',
        'y_pred_base = baseline.predict(x_test_tfidf)\n',
        'f1_base = f1_score(y_test, y_pred_base, average="weighted", zero_division=0)\n',
        'print(f"Baseline (default LR):  F1-weighted = {f1_base:.4f}")\n',
        '\n',
        '# --- Подбор гиперпараметров ---\n',
        'param_grid = {\n',
        '    "C": [0.1, 1, 10],\n',
        '    "class_weight": ["balanced", None],\n',
        '}\n',
        '\n',
        'search = HalvingGridSearchCV(\n',
        '    LogisticRegression(max_iter=500, random_state=42, penalty="l2", solver="lbfgs"),\n',
        '    param_grid,\n',
        '    cv=3,\n',
        '    scoring="f1_weighted",\n',
        '    factor=2,\n',
        '    verbose=1,\n',
        '    n_jobs=-1,\n',
        ')\n',
        'search.fit(x_train_tfidf, y_train)\n',
        '\n',
        'print(f"Best params: {search.best_params_}")\n',
        'print(f"Best CV score:  {search.best_score_:.4f}")\n',
        '\n',
        '# --- Лучшая модель ---\n',
        'best_clf = search.best_estimator_\n',
        'y_pred_best = best_clf.predict(x_test_tfidf)\n',
        'f1_best = f1_score(y_test, y_pred_best, average="weighted", zero_division=0)\n',
        'print(f"Best tuned:          F1-weighted = {f1_best:.4f}")\n',
        '\n',
        '# Сводка всех попыток\n',
        'cv_results = pd.DataFrame(search.cv_results_)[["param_C", "param_class_weight", "mean_test_score", "std_test_score", "rank_test_score"]]\n',
        'cv_results = cv_results.sort_values("rank_test_score")\n',
        'print("\\nGrid search results (top-6):")\n',
        'print(cv_results.to_string(index=False))\n',
    ]
}

with open('testTask/notebooks/pipeline_with_clustering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Updated')
