import sys
import os
from pathlib import Path

# Add the 'src' directory to sys.path so 'blenny' can be imported
# Hugging Face installs requirements, but we want to ensure the local 'src'
# is used if the package isn't installed in the environment.
root_dir = Path(__file__).parent
sys.path.append(str(root_dir / "src"))

# Streamlit apps are executed as scripts. 
# We just need to import the gui.app logic.
import gui.app
