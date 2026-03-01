"""Logging utilities for training and inference scripts."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


def setup_log_file(
    log_dir: Path,
    script_name: str,
    suffix: str = "",
) -> tuple[TextIO, Path]:
    """
    Create a log file with timestamp and return file handle and path.
    
    Args:
        log_dir: Directory where log files will be saved
        script_name: Name of the script (e.g., 'entrenar_ecapa', 'inferir')
        suffix: Optional suffix to add to filename (e.g., 'xvector', '05seg')
    
    Returns:
        Tuple of (file_handle, file_path)
    
    Usage:
        log_file, log_path = setup_log_file(Path('logs'), 'entrenar_ecapa', 'xvector')
        print(f"Logs saved to: {log_path}")
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_str = f"_{suffix}" if suffix else ""
    log_filename = f"{script_name}{suffix_str}_{timestamp}.log"
    log_path = log_dir / log_filename
    
    # Open file in append mode
    log_file = open(log_path, "w", encoding="utf-8")
    return log_file, log_path


class DualWriter:
    """Write to both console and log file simultaneously."""
    
    def __init__(self, console: TextIO, log_file: TextIO):
        """
        Initialize dual writer.
        
        Args:
            console: Console file handle (usually sys.stdout or sys.stderr)
            log_file: Log file handle
        """
        self.console = console
        self.log_file = log_file
    
    def write(self, message: str) -> None:
        """Write to both console and log file."""
        self.console.write(message)
        self.console.flush()
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self) -> None:
        """Flush both streams."""
        self.console.flush()
        self.log_file.flush()
    
    def close(self) -> None:
        """Close the log file (console stays open)."""
        self.log_file.close()


def redirect_output_to_log(
    log_dir: Path,
    script_name: str,
    suffix: str = "",
) -> tuple[DualWriter, Path]:
    """
    Redirect stdout and stderr to log file while keeping console output.
    
    Args:
        log_dir: Directory where log files will be saved
        script_name: Name of the script
        suffix: Optional suffix for the filename
    
    Returns:
        Tuple of (DualWriter instance, log_file_path)
    
    Usage:
        log_writer, log_path = redirect_output_to_log(
            Path('logs'), 'entrenar_ecapa', 'xvector'
        )
        sys.stdout = log_writer
        print("This goes to both console and log file")
        log_writer.close()  # Close when done
    """
    log_file, log_path = setup_log_file(log_dir, script_name, suffix)
    dual_writer = DualWriter(sys.stdout, log_file)
    return dual_writer, log_path
