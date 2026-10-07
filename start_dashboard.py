"""
Start het dashboard zonder iets in de terminal te typen:
open dit bestand in VS Code en klik op 'Run Python File' (het driehoekje rechtsboven).
Het dashboard staat daarna op http://localhost:8501
"""
import subprocess
import sys
from pathlib import Path

map_van_project = Path(__file__).parent
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"],
    cwd=map_van_project,
)
