"""Put the project root on sys.path, so that the tests import the two program files directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
