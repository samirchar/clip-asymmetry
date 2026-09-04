#Add path to src as a constant
from pathlib import Path
# Dynamically determine the root directory of the package
PACKAGE_PATH = Path(__file__).resolve().parent

#Architectures path
ARCHITECTURES_PATH = PACKAGE_PATH / "architectures"

# Project/repo root (useful in development)
REPO_PATH = PACKAGE_PATH.parent

# Default repo config dir (development)
REPOSITORY_CONFIG_DIR = REPO_PATH / "configs"
