"""Entry point for Streamlit Community Cloud: runs the chat UI in embedded mode."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "ui" / "app.py"), run_name="__main__")
