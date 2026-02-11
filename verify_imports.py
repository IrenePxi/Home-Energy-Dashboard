import sys
from pathlib import Path

# Add the project root to sys.path just in case, 
# although running from the root should be enough.
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print(f"Checking imports in {project_root}...")

try:
    import ui.tabs.forecasts
    print("[OK] ui.tabs.forecasts imported")
    import ui.tabs.dt_real
    print("[OK] ui.tabs.dt_real imported")
    import ui.tabs.scheduling_ui
    print("[OK] ui.tabs.scheduling_ui imported")
    import services.mqtt_manager
    print("[OK] services.mqtt_manager imported")
    import services.paths
    print("[OK] services.paths imported")
    import data_sources.electricity_prices
    print("[OK] data_sources.electricity_prices imported")
    import data_sources.pv
    print("[OK] data_sources.pv imported")
    import data_sources.weather
    print("[OK] data_sources.weather imported")
    import data_sources.co2
    print("[OK] data_sources.co2 imported")
    
    print("\nAll imports successful!")
except Exception as e:
    print(f"\n[ERROR] Import failed: {e}")
    sys.exit(1)
