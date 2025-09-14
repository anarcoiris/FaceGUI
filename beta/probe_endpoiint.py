# probe_endpoint.py
from core.core_face import probe_endpoint
import os, json

cfg = {
    "endpoint": os.environ.get("FACE_ENDPOINT"),
    "key": os.environ.get("FACE_KEY"),
    "auth_method": os.environ.get("AUTH_METHOD", "Subscription Key")
}

if __name__ == "__main__":
    print(json.dumps(probe_endpoint(cfg), indent=2))
