
import os
import json
import time
from pathlib import Path
from services.paths import results_dir

# File to store active sessions
# Using results_dir as it is a writable location
ACTIVE_USERS_FILE = results_dir() / "active_users.json"

def log_heartbeat(session_id: str, timeout_seconds: int = 300) -> int:
    """
    Updates the heartbeat for the given session_id.
    Removes sessions older than timeout_seconds.
    Returns the count of active users.
    """
    now = time.time()
    data = {}
    
    # Ensure results dir exists
    results_dir().mkdir(exist_ok=True)

    # Load existing data
    if ACTIVE_USERS_FILE.exists():
        try:
            with open(ACTIVE_USERS_FILE, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    
    # Update current session
    data[session_id] = now
    
    # Filter out old sessions
    active_sessions = {
        sid: timestamp 
        for sid, timestamp in data.items() 
        if now - timestamp < timeout_seconds
    }
    
    # Save back to file
    try:
        with open(ACTIVE_USERS_FILE, "w") as f:
            json.dump(active_sessions, f)
    except OSError:
        pass # Ignore write errors (e.g. concurrency race)
        
    return len(active_sessions)
