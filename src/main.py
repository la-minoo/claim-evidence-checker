"""
main.py  —  server entry point for Phase 4b
Imports infer (registers all core routes) then retrieval (registers /retrieve),
then starts uvicorn. This avoids the circular-import problem that occurs when
infer.py tries to import retrieval.py while retrieval.py imports from infer.py.

Run with:
    conda activate claimcheck
    cd C:\\Users\\1\\buss305-project\\claim-evidence-checker
    python src\\main.py

Port:
    Local: 8000 (default)
    HuggingFace Spaces: 7860 (set via PORT environment variable)
"""

import os
import sys
from pathlib import Path

# Ensure src\ is on the path (needed when run from project root)
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

# 1. Load model, define app, register all core routes
from infer import app, predict   # noqa: F401

# 2. Register /retrieve — retrieval.py does `from infer import app`
#    and decorates @app.post("/retrieve"). Importing it here, after infer
#    is fully loaded, completes that registration cleanly.
import retrieval  # noqa: F401

# 3. Start server
if __name__ == "__main__":
    import uvicorn

    # Read port from environment — 7860 for HF Spaces, 8000 locally
    port = int(os.getenv("PORT", 8000))

    print("\n[main] All routes registered:")
    for route in app.routes:
        if hasattr(route, "methods"):
            print(f"  {list(route.methods)[0]:6s} {route.path}")

    print(f"\n[main] Starting server on http://0.0.0.0:{port}")
    print("[main] Docs: http://localhost:{port}/docs")
    print("[main] Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
