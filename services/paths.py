import os
import sys
from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root directory."""
    # If frozen with PyInstaller (if ever used)
    if getattr(sys, 'frozen', False):
         return Path(sys._MEIPASS)
    
    # This file is in DT_dashboard/services/paths.py
    # Root is up 2 levels: services -> DT_dashboard (Project Root)
    return Path(__file__).resolve().parent.parent

def rpath(rel: str) -> Path:
    """Returns absolute path for a project-relative path.
    Args:
        rel: Relative path from project root (e.g., 'DT_dashboard/overview_live.html')
    """
    # Fix backslashes to forward slashes for Path compatibility if strict
    # but Path handles both on Windows usually.
    return get_project_root() / rel

def results_dir() -> Path:
    return rpath("results")
