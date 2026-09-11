"""Logging for the API and the worker, from one file.

uvicorn reads log_config.json directly (--log-config); arq needs an import
path to a dict (--custom-log-dict backend.log_config.LOG_CONFIG).
"""

import json
from pathlib import Path

LOG_CONFIG = json.loads((Path(__file__).parent / "log_config.json").read_text())
