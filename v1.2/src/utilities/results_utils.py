import json
from pathlib import Path


def save_json(data, path):
    """Save a dictionary as a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(data, file, indent=2)


def load_json(path):
    """Load a JSON file and return its contents."""
    with open(path) as file:
        return json.load(file)
