from __future__ import annotations

import sys
from pathlib import Path

# Vercel installs declared dependencies but does not always add a src-layout package
# to sys.path while packaging a Python function.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interview_coach.api.app import create_app  # noqa: E402

app = create_app(serve_frontend=False)
