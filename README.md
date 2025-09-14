Azure Face GUI — Final package
==============================

Contenido:
- core/core_face.py  -> SDK-first Face API helpers + probe
- core/core_storage.py -> Blob helpers (uses user's version)
- core/db.py -> sqlite helpers
- ui/streamlit_app.py -> Streamlit UI (Config, Probe, Detect, Verify, Attributes, Collections, Storage, Batch, Liveness, Logs)
- api.py -> FastAPI proxy endpoints (detect, identify)
- probe_endpoint.py -> CLI probe for endpoint capabilities
- streamlit_azure_face_gui.py -> launcher (imports UI)
- requirements.txt, .env.sample

Usage:
1. Install dependencies:
   pip install -r requirements.txt

2. Set environment variables or use Streamlit Config/.env:
   - FACE_ENDPOINT and FACE_KEY (or use Azure AD with DefaultAzureCredential)

3. Run UI:
   streamlit run streamlit_azure_face_gui.py

4. (Optional) Run API proxy:
   uvicorn api:app --reload --port 8000

Notes:
- The app contains a probe that checks which Face API features your endpoint supports and adapts the UI accordingly.
- Some Face API features (LargePersonGroup, attributes, liveness) may require an approved Face resource in Azure.
- Keep keys secure. Prefer Managed Identity in production.
