"""ClaimSense AI — API server entrypoint.

Run the server
--------------
    # Development (auto-reload on file changes)
    python main.py

    # Or directly with uvicorn
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

API docs (once the server is running)
--------------------------------------
    Swagger UI  →  http://localhost:8000/docs
    ReDoc       →  http://localhost:8000/redoc
    Health      →  http://localhost:8000/health
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # hot-reload for development
        log_level="info",
    )
