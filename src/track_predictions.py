import os
import sys 
import pandas as pd
from sklearn.metrics import log_loss

MATCHES = 'data/powerbi/matches.csv'
PREDS = 'data/powerbi/predictions.csv'
LOG = 'data/processed/predictions_log.csv'
KEY = ['season', 'home_team_id', 'away_team_id']
DAYS_AHEAD = 7
RESULT_COLS = ['home_goals', 'away_goals', 'result']
PROBS = ['p_home', 'p_draw', 'p_away']

def freeze():
    """Add upcoming matches to the log. Existing rows are never touched."""
    matches = pd.read_csv(MATCHES)
    matches['date'] = pd.to_datetime(matches['date'], utc=True)
    preds = pd.read_csv(PREDS)

    now = pd.Timestamp.now(tz='UTC')
    upcoming = matches[
        (~matches['played']) 
        & (matches['date'] > now)
        & (matches['date'] <= now + pd.Timedelta(days=DAYS_AHEAD))
        
    ]
    new = upcoming.merge(preds, on="match_id")[KEY + ["date"] + PROBS]
    
    if os.path.exists(LOG):
        old = pd.read_csv(LOG)
        new = new.merge(old[KEY], on='KEY', how='left', indicator=True)
        new = new[new['_merge' == 'left_only']].drop(columns='merge')
    else:
        old = pd.DataFrame()

    new['logged_at'] = now.isoformat()
    for col in RESULT_COLS:
        new[col] = None

    pd.concat([old, new]).to_csv(LOG, index=False)


def score():
    """Fill in results for logged matches that have been played, then report metrics."""
    log = pd.read_csv(LOG)
    matches = pd.read_csv(MATCHES)
    played = matches[matches['played']][KEY + RESULT_COLS]

    scored = log[log['result'].notna()]
    pending = log[log['result'].isna()].drop(columns=RESULT_COLS)
    pending = pending.merge(played, on=KEY, how='left')

    log = pd.concat([scored, pending]).sort_values('date')
    log.to_csv(LOG, index=False)

    done = log.dropna(subset=['result'])
    if done.empty:
        print('No scored matches yet')
        return
    y = done['result'].map({"H": 0, "D": 1, "A": 2})
    probs = done[PROBS].values
    print(f"Scored matches: {len(done)} of {len(log)}")
    print(f"Log loss: {log_loss(y, probs, labels=[0, 1, 2]):.4f}")
    print(f"Accuracy: {(probs.argmax(axis=1) == y).mean():.4f}")


if __name__ == '__main__':
    commands = {'log': freeze, 'score': score}
    commands[sys.argv[1]]()
