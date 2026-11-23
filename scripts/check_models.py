import json
from backend.engine import engine

status = engine.status()
print(json.dumps(status, indent=2))
