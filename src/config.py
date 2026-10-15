import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'

LEAGUE = 'PL'

# CSV (https://datahub.io/football/english-premier-league): Manually downloaded
# Season code: "2122" = 2021-22
CSV_SEASONS = ['2122', '2223', '2324', '2425', '2526']

# API (https://www.football-data.org/): On-going season
API_TOKEN = os.getenv('FOOTBALL_DATA_TOKEN')
API_BASE_URL = 'https://api.football-data.org/v4'
API_LEAGUE = LEAGUE
API_SEASONS = [2026]

PAUSE_SECONDS = 7
MAX_RETRIES = 3