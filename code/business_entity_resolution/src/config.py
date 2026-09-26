import os

ROOT = os.environ.get("BER_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DATA_DIR = os.environ.get("BER_DATA", os.path.join(ROOT, "data", "dataset"))
WORK_DIR = os.environ.get("BER_WORK", os.path.join(ROOT, "work"))
OUT_DIR = os.environ.get("BER_OUT", os.path.join(ROOT, "output"))
SEED = 42
