import yaml
import os
from pathlib import Path
def load_config(config_path="configs/config.yaml"):

    print("\n==========================================")
    print("        LOADING CONFIGURATION FILE        ")
    print("==========================================")
    print(f"Target Path: {os.path.abspath(config_path)}")
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("✓ Configuration successfully loaded!")
    print(f"Total Top-Level Keys Found: {len(config.keys())}")
    
    print("\n--- Loaded Configuration Contents ---")
    for key, value in config.items():
        if isinstance(value, list):
            print(f"  • {key}: List with {len(value)} items -> {value[:3]}...")
        else:
            print(f"  • {key}: {value}")
            
    print("==========================================\n")
    return config


if __name__ == "__main__":
    config_data = load_config("configs/config.yaml")