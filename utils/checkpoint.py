"""Checkpoint/pause-resume utilities for training scripts.

The state file is stored as `.training_state.json` inside the models directory.
When training completes successfully, the state file is automatically deleted.

Usage in a training script:
    from utils.checkpoint import TrainingCheckpoint, setup_pause_handler, pause_requested

    setup_pause_handler()  # Installs SIGINT → graceful pause handler

    ckpt = TrainingCheckpoint(models_dir)
    state = ckpt.load()

    if state and args.resume:
        start_fold = len(state["completed_folds"])
        fold_metrics         = state["fold_results"]
        fold_best_epochs     = state["fold_best_epochs"]
        fold_training_times  = state["fold_training_times"]
        all_fold_histories   = state["fold_histories"]
        resumed_from_fold    = start_fold
        pause_count          = state.get("pause_count", 0)
        print(f"[RESUME] Resuming from fold {start_fold}/{total_folds}")
    else:
        state = ckpt.initialize()
        start_fold = 0
        fold_metrics, fold_best_epochs, fold_training_times, all_fold_histories = [], [], [], []
        resumed_from_fold = None
        pause_count = 0

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        if fold_idx < start_fold:
            continue
        # ... train ...
        ckpt.save_fold(state, fold_idx, metrics, fold_time, best_epoch, fold_history)
        if pause_requested():
            ckpt.mark_paused(state)
            print(f"[PAUSE] Paused after fold {fold_idx + 1}. Re-run with --resume to continue.")
            sys.exit(0)

    # After JSON save:
    ckpt.delete()
"""

import json
import sys
import signal
from pathlib import Path
from datetime import datetime

_pause_requested = False


def setup_pause_handler() -> None:
    """Install a SIGINT handler that requests a graceful pause after the current fold."""
    def _handler(sig, frame):
        global _pause_requested
        if not _pause_requested:
            print("\n[PAUSE] Pause requested. Will stop after current fold completes.")
            print("        Re-run with --resume to continue from where you left off.")
            _pause_requested = True
        else:
            print("\n[PAUSE] Force-quit requested. Exiting immediately (fold result lost).")
            sys.exit(1)
    signal.signal(signal.SIGINT, _handler)


def pause_requested() -> bool:
    """Return True if the user has requested a graceful pause (Ctrl-C)."""
    return _pause_requested


class TrainingCheckpoint:
    """Manages per-fold checkpoint state for pause/resume capability.

    The state file is stored as ``models_dir/.training_state.json``.
    Training time is measured as the *sum of individual fold times*, so any
    time spent paused between runs is automatically excluded.
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self.state_file = self.models_dir / ".training_state.json"

    def exists(self) -> bool:
        return self.state_file.exists()

    def load(self) -> dict | None:
        """Return saved state dict, or None if no checkpoint exists."""
        if not self.state_file.exists():
            return None
        try:
            return json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def initialize(self) -> dict:
        """Create a fresh checkpoint and return the initial state dict."""
        state: dict = {
            "completed_folds": [],
            "fold_results": [],
            "fold_training_times": [],
            "fold_best_epochs": [],
            "fold_histories": [],
            "was_paused": False,
            "pause_count": 0,
            "resumed_from_fold": None,
            "start_timestamp": datetime.now().isoformat(),
        }
        self._write(state)
        return state

    def save_fold(
        self,
        state: dict,
        fold_idx: int,
        fold_result: dict,
        fold_time: float,
        best_epoch: int | None = None,
        history: list | None = None,
    ) -> None:
        """Append a completed fold to the checkpoint and persist it to disk."""
        state["completed_folds"].append(fold_idx)
        state["fold_results"].append(fold_result)
        state["fold_training_times"].append(round(fold_time, 2))
        if best_epoch is not None:
            state["fold_best_epochs"].append(best_epoch)
        if history is not None:
            state["fold_histories"].append(history)
        self._write(state)

    def mark_paused(self, state: dict) -> None:
        """Record that the run was interrupted (sets was_paused=True)."""
        state["was_paused"] = True
        state["pause_count"] = state.get("pause_count", 0) + 1
        self._write(state)

    def delete(self) -> None:
        """Remove the checkpoint file after a successful full run."""
        if self.state_file.exists():
            self.state_file.unlink()

    def _write(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, indent=2))
