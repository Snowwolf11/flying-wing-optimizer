"""Launch the interactive Plotly/Dash design GUI."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flyingwing.gui.app import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False, host="127.0.0.1", port=8050)
