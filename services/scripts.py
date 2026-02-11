import sys
import subprocess
import os
from .paths import rpath, get_project_root

def run_py(rel_path: str, cwd_rel: str = None, check: bool = False):
    """
    Run a python script in a subprocess.
    Args:
        rel_path: path to script relative to project root (e.g. "Electricity_price/ElepredictionML.py")
                  If it ends with .py, it runs as script. If not, could be treated as module if needed,
                  but for this project we generally point to .py files.
        cwd_rel: (optional) working directory relative to project root.
        check: if True, raises CalledProcessError on non-zero exit code (if not captured).
               However, since we capture output, we return returncode and let caller decide.
    
    Returns:
        (returncode, stdout, stderr)
    """
    script_path = rpath(rel_path)
    cwd = rpath(cwd_rel) if cwd_rel else get_project_root()
    
    # Check if we should run as module or script. 
    # Current codebase mostly runs absolute script paths.
    cmd = [sys.executable, str(script_path)]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False # We handle returncode manually
        )
        return result.returncode, result.stdout, result.stderr
        
    except Exception as e:
        return -1, "", str(e)

def start_script_detached(rel_path, cwd_rel_path=None):
    """Start a script in detached mode (real-time loops)."""
    script_path = rpath(rel_path)
    cwd = rpath(cwd_rel_path) if cwd_rel_path else get_project_root()
    
    if os.name == "nt":
        subprocess.Popen(
            [sys.executable, str(script_path)], 
            cwd=cwd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    else:
        subprocess.Popen(
            [sys.executable, str(script_path)], 
            cwd=cwd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
