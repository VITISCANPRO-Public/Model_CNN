#                                          LIBRARIES IMPORT
# ================================================================================================

import yaml

#                                           CONFIGURATION
# ================================================================================================

def load_config(config_path: str = "config.yml") -> dict:
    """
    Loads the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from: {config_path}")
    return config