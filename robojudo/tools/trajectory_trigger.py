from __future__ import annotations

import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)

TRAJECTORY_START_FILE_ENV = "ROBOJUDO_TRAJECTORY_START_FILE"


def request_trajectory_start() -> bool:
    """Request recording without coupling the policy to recorder internals."""
    trigger_path = os.environ.get(TRAJECTORY_START_FILE_ENV)
    if not trigger_path:
        return False
    try:
        Path(trigger_path).touch(exist_ok=True)
    except OSError as exc:
        logger.error("Could not request trajectory recording: %s", exc)
        return False
    return True
