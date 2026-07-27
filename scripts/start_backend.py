import os
import sys
from pathlib import Path

# Ensure project root on sys.path for clean
# package resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Run uvicorn from project root using
# package-qualified app path
os.chdir(PROJECT_ROOT)

try:
    import uvicorn
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"uvicorn import failed: {exc}")

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT / "backend")],
    )
