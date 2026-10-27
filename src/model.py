import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss

from config import PROCESSED_DIR, ROOT
from features import FEATURE_COLS

OUT = ROOT / 'data' / 'powerbi'

# Temporal cutoffs: everything before VAL_START is train.
VAL_START = '2024-07-01'
TEST_START = '2025-07-01'
LABELS = ['H', 'D', 'A']     # class 0, 1, 2
PROB_COLS = ['p_home', 'p_draw', 'p_away']

SEED = 42


def load_data():
    '''Read the features and label every match with its split.'''
    df = pd.read_csv(PROCESSED_DIR / 'features.csv', parse_dates=['date'])

    df['split'] = np.select(
        [~df['played'].astype(bool),
         df['date'] < VAL_START,
         df['date'] < TEST_START],
        ['pending', 'train', 'val'],
        default='test',
    )

    df['y'] = df['result'].map({'H': 0, 'D': 1, 'A': 2}).fillna(-1).astype(int)
    assert (df.loc[df['split'] != 'pending', 'y'] >= 0).all(), 'Played match without result'
    return df


def get_models():
    '''Baseline and candidate models, all unfitted.'''
    return {
        'Dummy Prior': DummyClassifier(strategy='prior', random_state=SEED),
        'Dummy Uniform': DummyClassifier(strategy='uniform', random_state=SEED),
        'Dummy Stratified': DummyClassifier(strategy='stratified'),
        'Dummy Most Frequent': DummyClassifier(strategy='most_frequent'),
        'Dummy Constant=1': DummyClassifier(strategy='constant', constant=1),
        'Dummy Constant=2': DummyClassifier(strategy='constant', constant=2),
        'LR Base': make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=SEED, solver='lbfgs')
        ),
        'LR Balanced': make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=SEED, solver='lbfgs', class_weight='balanced')
        ),
        'RF Base': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=20,
            class_weight=None, n_jobs=-1, random_state=SEED
        ),
        'RF Balanced': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight='balanced', n_jobs=-1, random_state=SEED
        ),
        'XGB Base': XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            eval_metric='mlogloss', random_state=SEED, n_jobs=-1,
        ),
        'XGB Regularized': XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=2,
            eval_metric='mlogloss', random_state=SEED, n_jobs=-1
        )
    }


def train_models(df, models):
    '''Train on train and compare models on validation. Sorted by Log-Loss.'''
    tr = df['split'] == 'train'
    va = df['split'] == 'val'

    x_train = df.loc[tr, FEATURE_COLS]
    x_val = df.loc[va, FEATURE_COLS]
    y_train = df.loc[tr, 'y']
    y_val = df.loc[va, 'y']

    results = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        proba = model.predict_proba(x_val)

        results[name] = {
            'Log-Loss': log_loss(y_val, proba, labels=[0, 1, 2]),
            'Accuracy': accuracy_score(y_val, pred)
        }

    return pd.DataFrame(results).T.round(4).sort_values('Log-Loss')


def build_predictions(df, model):
    '''Generate out-of-sample probabilities for val, test and pending matches.'''
    pred = df[['match_id', 'split', 'y']].copy()

    for col in PROB_COLS:
        pred[col] = np.nan     # train rows stay empty on purpose

    plan = {
        'val': ['train'],
        'test': ['train', 'val'],
        'pending': ['train', 'val', 'test']
    }

    X, y = df[FEATURE_COLS], df['y']

    for split, fit_splits in plan.items():
        fit = df['split'].isin(fit_splits)
        mask = df['split'] == split

        if mask.any():
            model.fit(X[fit], y[fit])
            pred.loc[mask, PROB_COLS] = model.predict_proba(X[mask])

    return pred


def score(pred):
    '''Calculate metrics on validation and test predictions.'''
    results = {}

    for split in ['val', 'test']:
        data = pred[pred['split'] == split]
        y = data['y']
        proba = data[PROB_COLS].values

        results[split] = {
            'Log-Loss': log_loss(y, proba, labels=[0, 1, 2]),
            'Accuracy': accuracy_score(y, proba.argmax(axis=1))
        }

    return pd.DataFrame(results).T.round(4)


def calibration_table(pred, split, n_bins=5):
    '''Compare predicted probabilities with real frequencies.'''
    data = pred[pred['split'] == split]
    y = data['y'].values

    rows = []
    for k, label in enumerate(LABELS):
        tmp = pd.DataFrame({'pred': data[PROB_COLS[k]].values, 'real': (y == k).astype(int)})
        tmp['bin'] = pd.qcut(tmp['pred'], n_bins, labels=False, duplicates='drop') + 1
        table = tmp.groupby('bin').agg(
            matches=('real', 'size'),
            predicted=('pred', 'mean'),
            real=('real', 'mean')
        )
        table['error'] = (table['predicted'] - table['real']).abs()
        table['ece'] = table['error'] * table['matches'] / len(tmp)
        table.insert(0, 'class', label)

        rows.append(table.reset_index())

    return pd.concat(rows, ignore_index=True).round(3)


def export(df, pred, scores, calibration, best_model):
    '''Write the CSV tables that Power BI will read.'''
    OUT.mkdir(parents=True, exist_ok=True)

    teams = sorted(set(df['home_team']) | set(df['away_team']))
    team_id = {t: i for i, t in enumerate(teams, start=1)}
    pd.DataFrame({'team_id': list(team_id.values()), 'team': list(team_id.keys())}).to_csv(OUT / 'teams.csv', index=False)

    out = df.copy()
    out['home_team_id'] = out['home_team'].map(team_id)
    out['away_team_id'] = out['away_team'].map(team_id)
    cols_export = ['match_id', 'date', 'home_team_id', 'away_team_id', 'home_goals', 'away_goals', 'result', 'season', 'played', 'split']
    out[cols_export].to_csv(OUT / 'matches.csv', index=False)
    out[['match_id'] + FEATURE_COLS].to_csv(OUT / 'features.csv', index=False)

    labels = {'p_home': 'H', 'p_draw': 'D', 'p_away': 'A'}
    pred['pred_result'] = pred[PROB_COLS].dropna().idxmax(axis=1).map(labels)
    pred['model'] = best_model
    pred[['match_id'] + PROB_COLS + ['pred_result', 'model']].to_csv(OUT / 'predictions.csv', index=False)
    scores.rename_axis(['model', 'split']).reset_index().to_csv(OUT / 'metrics.csv', index=False)
    calibration.to_csv(OUT / 'calibration.csv', index=False)


def main():
    df = load_data()
    models = get_models()

    metrics = train_models(df, models)
    print('MODELS (validation)')
    print(metrics)
    print('-'*100)

    # Best real model: Dummy baselines can't be the pick
    candidates = metrics[~metrics.index.str.startswith('Dummy')]
    best = candidates.index[0]
    print('Best model on val: ', best)
    print('-'*100)

    pred = build_predictions(df, models[best])
    baseline = build_predictions(df, models['Dummy Prior'])

    scores = pd.concat({
        'Dummy Prior': score(baseline),
        best: score(pred)
    })

    print('MODEL SCORES')
    print(scores)
    print('-'*100)

    calibration = calibration_table(pred, split='test')
    print('TEST CALIBRATION')
    print(calibration.groupby('class')['ece'].sum().round(3))
    print('-'*100)

    export(df, pred, scores, calibration, best)


if __name__ == '__main__':
    main()