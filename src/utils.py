import os
import yaml
from typing import Dict, Any, List

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def format_sources(sources: List[str]) -> str:
    """Format a list of source paths for display."""
    formatted_sources = []
    for source in sources:
        filename = os.path.basename(source)
        formatted_sources.append(f"- {filename}")
    
    return "\n".join(formatted_sources)