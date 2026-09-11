# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

# pictomesh patch: messages go through the logging module (the app's timestamped format)
# instead of coloured print() calls.
_STDLIB = logging.getLogger("depth_anything_3")
_STDLIB_LEVELS = {"ERROR": logging.ERROR, "WARN": logging.WARNING, "INFO": logging.INFO, "DEBUG": logging.DEBUG}


class Color:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    WHITE = "\033[97m"
    GREEN = "\033[92m"
    RESET = "\033[0m"


LOG_LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}

COLOR_MAP = {"ERROR": Color.RED, "WARN": Color.YELLOW, "INFO": Color.WHITE, "DEBUG": Color.GREEN}


def get_env_log_level():
    level = os.environ.get("DA3_LOG_LEVEL", "INFO").upper()
    return LOG_LEVELS.get(level, LOG_LEVELS["INFO"])


class Logger:
    def __init__(self):
        self.level = get_env_log_level()

    def log(self, level_str, *args, **kwargs):
        level_key = level_str.split(":")[0].strip()
        level_val = LOG_LEVELS.get(level_key)
        if level_val is None:
            raise ValueError(f"Unknown log level: {level_str}")
        if self.level >= level_val:
            _STDLIB.log(_STDLIB_LEVELS[level_key], " ".join(str(arg) for arg in args))

    def error(self, *args, **kwargs):
        self.log("ERROR:", *args, **kwargs)

    def warn(self, *args, **kwargs):
        self.log("WARN:", *args, **kwargs)

    def info(self, *args, **kwargs):
        self.log("INFO:", *args, **kwargs)

    def debug(self, *args, **kwargs):
        self.log("DEBUG:", *args, **kwargs)


logger = Logger()

__all__ = ["logger"]

if __name__ == "__main__":
    logger.info("This is an info message")
    logger.warn("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
