from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.2

N_ESTIMATORS = 100

