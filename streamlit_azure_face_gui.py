#!/usr/bin/env python3
"""
Streamlit GUI para Azure Face API — multi-tab (refactor + FastAPI + Attributes tab)
Archivo: streamlit_azure_face_gui.py

Resumen: App en Streamlit que cubre Detect, Verify, Identify, FindSimilar,
Gestión de colecciones (LargePersonGroup / Persons / persistedFaces), Liveness y utilidades.

Cambios clave en esta versión:
- Añadida pestaña "Attributes" que permite seleccionar y visualizar todas las características faciales que Face API puede devolver (edad, género, emociones, hair color, accesorios, maquillaje, occlusion, blur, exposure, noise, etc.).
- Refactor: separación entre funciones "core" (independientes de Streamlit) y la UI.
- FastAPI app con endpoints `/detect` y `/identify` (proxy mínimo).
- Correcciones de keys de widgets para evitar duplicados.
- Compatibilidad para rerun entre distintas versiones de Streamlit mediante _maybe_rerun().
- Manejo explícito de errores 403 UnsupportedFeature con mensaje instructivo.
- Por defecto la UI no solicita faceId (para evitar 403 en recursos no aprobados); hay un flag en Config para habilitar las features de identificación si tu recurso está aprobado.

Uso:
- Ejecutar la UI: `streamlit run streamlit_azure_face_gui.py`
- Ejecutar la API: `uvicorn streamlit_azure_face_gui:app --port 8000`

Dependencias:
- pip install streamlit requests pillow pandas python-dotenv fastapi uvicorn python-multipart
- Si usas Azure AD: pip install azure-identity

Notas legales: el uso de biometría tiene implicaciones legales. Usa con consentimiento.
"""

import os, time, json, base64, requests, sqlite3, zipfile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO, TextIOWrapper
import pandas as pd

DB_FILE = os.path.join(os.getcwd(), 'face_gui.sqlite')
API_PREFIX = "/face/v1.0"
DEFAULT_RECOGNITION_MODEL = "recognition_04"
DEFAULT_DETECTION_MODEL = "detection_03"

# ---------------- DB UTILS ----------------
def init_db(db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS configs (
        name TEXT PRIMARY KEY,
        cfg_json TEXT,
        created_at TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

def save_config_to_db(name, cfg, db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('REPLACE INTO configs (name, cfg_json, created_at) VALUES (?, ?, ?)', 
                (name, json.dumps(cfg), time.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def list_saved_configs(db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('SELECT name, created_at FROM configs ORDER BY created_at DESC')
    rows = cur.fetchall()
    conn.close()
    return rows

def load_config_from_db(name, db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('SELECT cfg_json FROM configs WHERE name=?', (name,))
    row = cur.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def delete_config_from_db(name, db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('DELETE FROM configs WHERE name=?', (name,))
    conn.commit()
    conn.close()

# ---------------- HTTP UTILS ----------------
def make_requests_session(cfg=None):
    session = requests.Session()
    retries = cfg.get('retries') if cfg else 3
    backoff = cfg.get('backoff_factor') if cfg else 0.3
    if retries and retries > 0:
        retry = Retry(total=retries, backoff_factor=backoff,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET","POST","PUT","DELETE","HEAD","OPTIONS"])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
    return session

def get_auth_headers_from_cfg(cfg):
    if cfg.get('auth_method') == 'Azure AD':
        try:
            from azure.identity import DefaultAzureCredential
        except Exception:
            raise RuntimeError('Instala azure-identity: pip install azure-identity')
        cred = DefaultAzureCredential()
        token = cred.get_token('https://cognitiveservices.azure.com/.default')
        return {'Authorization': f'Bearer {token.token}'}
    else:
        key = cfg.get('key') or os.environ.get('FACE_KEY')
        if not key:
            raise ValueError('Subscription Key no configurada (cfg.key o env FACE_KEY)')
        return {'Ocp-Apim-Subscription-Key': key}

def api_url_from_cfg(cfg, path):
    endpoint = cfg.get('endpoint') or os.environ.get('FACE_ENDPOINT')
    if not endpoint:
        raise ValueError('Endpoint no configurado (cfg.endpoint o env FACE_ENDPOINT)')
    return f"{endpoint.rstrip('/')}" + API_PREFIX + path


def do_request_core(cfg, method, path, params=None, headers=None, data=None, json_payload=None, stream=False):
    session = make_requests_session(cfg)
    url = api_url_from_cfg(cfg, path)
    hdrs = {**(headers or {}), **get_auth_headers_from_cfg(cfg)}
    resp = session.request(method, url, params=params, headers=hdrs, data=data, 
                           json=json_payload, timeout=cfg.get('timeout',10), stream=stream)
    # Manejo específico para 403 UnsupportedFeature (mensaje más útil)
    if resp.status_code == 403:
        try:
            body = resp.json()
            # intenta extraer inner error
            inner = None
            if isinstance(body, dict):
                inner = body.get('error', {}) or body
                code = inner.get('code') if isinstance(inner, dict) else None
                msg = inner.get('message') if isinstance(inner, dict) else str(inner)
                if code and 'UnsupportedFeature' in str(code):
                    hint = ("Azure Face API: UnsupportedFeature. Las capacidades de identificación/verification/person-groups pueden estar vetadas en tu recurso. "
                            "Si necesitas esas features, solicita acceso en: aka.ms/facerecognition. "
                            "Si solo necesitas detección/atributos, configura la app para no solicitar faceId (Config -> deshabilitar identificación).")
                    raise RuntimeError(f"403 UnsupportedFeature: {msg} — {hint}")
                else:
                    raise RuntimeError(f"403 Forbidden: {body}")
            else:
                resp.raise_for_status()
        except ValueError:
            resp.raise_for_status()
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return resp.text

# ---------------- FACE API CORE ----------------
def detect_faces_core(cfg, img_bytes, return_landmarks=True, return_face_id=True, 
                      detection_model=DEFAULT_DETECTION_MODEL, recognition_model=DEFAULT_RECOGNITION_MODEL, 
                      return_face_attributes=None):
    params = {
        'returnFaceId': str(return_face_id).lower(),
        'returnFaceLandmarks': str(return_landmarks).lower(),
        'detectionModel': detection_model,
        'recognitionModel': recognition_model
    }
    if return_face_attributes:
        params['returnFaceAttributes'] = return_face_attributes
    return do_request_core(cfg, 'POST', '/detect', params=params, 
                           headers={'Content-Type': 'application/octet-stream'}, data=img_bytes)

def verify_faces_core(cfg, faceId1, faceId2):
    return do_request_core(cfg, 'POST', '/verify', 
                           headers={'Content-Type': 'application/json'}, 
                           json_payload={'faceId1': faceId1, 'faceId2': faceId2})

def identify_core(cfg, faceIds, largePersonGroupId, maxNumOfCandidatesReturned=5, confidenceThreshold=0.5):
    return do_request_core(cfg, 'POST', '/identify', 
                           headers={'Content-Type': 'application/json'}, 
                           json_payload={'faceIds': faceIds,'largePersonGroupId': largePersonGroupId,
                                         'maxNumOfCandidatesReturned': maxNumOfCandidatesReturned,
                                         'confidenceThreshold': confidenceThreshold})

def find_similar_core(cfg, faceId, largeFaceListId=None, faceIds=None, maxNumOfCandidatesReturned=5, mode='matchPerson'):
    payload = {'faceId': faceId, 'maxNumOfCandidatesReturned': maxNumOfCandidatesReturned, 'mode': mode}
    if largeFaceListId: payload['largeFaceListId'] = largeFaceListId
    if faceIds: payload['faceIds'] = faceIds
    return do_request_core(cfg, 'POST', '/findsimilars', 
                           headers={'Content-Type': 'application/json'}, 
                           json_payload=payload)

# --- LargePersonGroup (abreviado, pero completo en tu versión anterior) ---
def create_large_person_group_core(cfg, group_id, name, recognition_model=DEFAULT_RECOGNITION_MODEL):
    return do_request_core(cfg, 'PUT', f'/largepersongroups/{group_id}', 
                           headers={'Content-Type': 'application/json'}, 
                           json_payload={'name': name, 'recognitionModel': recognition_model})

def list_large_person_groups_core(cfg):
    return do_request_core(cfg, 'GET', '/largepersongroups')

def create_person_core(cfg, group_id, person_name, user_data=None):
    payload = {'name': person_name}
    if user_data: payload['userData'] = user_data
    return do_request_core(cfg, 'POST', f'/largepersongroups/{group_id}/persons', 
                           headers={'Content-Type': 'application/json'}, json_payload=payload)

def add_face_to_person_core(cfg, group_id, person_id, img_bytes):
    return do_request_core(cfg, 'POST', f'/largepersongroups/{group_id}/persons/{person_id}/persistedFaces', 
                           headers={'Content-Type': 'application/octet-stream'}, data=img_bytes)

def train_large_person_group_core(cfg, group_id):
    return do_request_core(cfg, 'POST', f'/largepersongroups/{group_id}/train')

def get_training_status_core(cfg, group_id):
    return do_request_core(cfg, 'GET', f'/largepersongroups/{group_id}/training')

def list_persons_in_group_core(cfg, group_id, top=100, skip=0):
    return do_request_core(cfg, 'GET', f'/largepersongroups/{group_id}/persons?top={top}&skip={skip}')

def get_person_core(cfg, group_id, person_id):
    return do_request_core(cfg, 'GET', f'/largepersongroups/{group_id}/persons/{person_id}')

def delete_persisted_face_core(cfg, group_id, person_id, persistedFaceId):
    session = make_requests_session(cfg)
    url = api_url_from_cfg(cfg, f'/largepersongroups/{group_id}/persons/{person_id}/persistedFaces/{persistedFaceId}')
    resp = session.delete(url, headers=get_auth_headers_from_cfg(cfg), timeout=cfg.get('timeout',10))
    resp.raise_for_status()
    return True

# ---------------- LIVENESS CORE ----------------
def create_liveness_session_core(cfg, mode='active', verify_image_bytes=None):
    payload = {'mode': mode}
    if verify_image_bytes:
        payload['verifyImage'] = {'data': base64.b64encode(verify_image_bytes).decode('utf-8')}
    return do_request_core(cfg, 'POST', '/detectLiveness-sessions', 
                           headers={'Content-Type': 'application/json'}, json_payload=payload)

def get_liveness_session_core(cfg, session_id):
    return do_request_core(cfg, 'GET', f'/detectLiveness-sessions/{session_id}')

# ---------------- FASTAPI PROXY ----------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title='Face API Proxy (minimal)')

class Creds(BaseModel):
    endpoint: Optional[str] = None
    key: Optional[str] = None
    auth_method: Optional[str] = None

class DetectRequest(BaseModel):
    creds: Optional[Creds] = None
    image_base64: str
    returnFaceLandmarks: Optional[bool] = False
    returnFaceId: Optional[bool] = True
    detectionModel: Optional[str] = DEFAULT_DETECTION_MODEL
    recognitionModel: Optional[str] = DEFAULT_RECOGNITION_MODEL
    returnFaceAttributes: Optional[str] = None

class IdentifyRequest(BaseModel):
    creds: Optional[Creds] = None
    faceIds: List[str]
    largePersonGroupId: str
    maxNumOfCandidatesReturned: Optional[int] = 5
    confidenceThreshold: Optional[float] = 0.5

def build_cfg_from_creds(creds: Optional[Creds]):
    return {
        'endpoint': creds.endpoint if creds and creds.endpoint else os.environ.get('FACE_ENDPOINT'),
        'key': creds.key if creds and creds.key else os.environ.get('FACE_KEY'),
        'auth_method': creds.auth_method if creds and creds.auth_method else 'Subscription Key',
        'timeout': 10,'retries': 2,'backoff_factor': 0.3
    }

@app.post('/detect')
def api_detect(req: DetectRequest):
    cfg = build_cfg_from_creds(req.creds)
    try:
        img_bytes = base64.b64decode(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid base64: {e}')
    try:
        res = detect_faces_core(cfg, img_bytes, return_landmarks=req.returnFaceLandmarks,
                                return_face_id=req.returnFaceId, detection_model=req.detectionModel,
                                recognition_model=req.recognitionModel, return_face_attributes=req.returnFaceAttributes)
        return {'result': res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/identify')
def api_identify(req: IdentifyRequest):
    cfg = build_cfg_from_creds(req.creds)
    try:
        res = identify_core(cfg, req.faceIds, req.largePersonGroupId, 
                            maxNumOfCandidatesReturned=req.maxNumOfCandidatesReturned, 
                            confidenceThreshold=req.confidenceThreshold)
        return {'result': res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- STREAMLIT COMPATIBILITY ----------------
def _maybe_rerun():
    """Compatibility helper for Streamlit rerun across versions.
    Tries st.experimental_rerun (older), then st.rerun (newer).
    Raises RuntimeError if neither exists to make the failure explicit.
    """
    try:
        import streamlit as _st
    except Exception:
        raise RuntimeError('streamlit no disponible en el entorno al llamar a _maybe_rerun()')
    if hasattr(_st, 'experimental_rerun'):
        return _st.experimental_rerun()
    if hasattr(_st, 'rerun'):
        return _st.rerun()
    raise RuntimeError(f"Tu versión de Streamlit ({getattr(_st, '__version__', 'desconocida')}) no soporta rerun programático.")

# ---------------- STREAMLIT UI ----------------
def run_streamlit():
    import streamlit as st
    from PIL import Image, ImageDraw, ImageFont

    st.write("Streamlit GUI para Azure Face API lista. Ejecuta con: `streamlit run streamlit_azure_face_gui.py`")

    # safe secret helper
    def safe_get_secret_streamlit(key, default=''):
        try:
            return st.secrets.get(key, default)
        except Exception:
            try:
                return st.secrets[key]
            except Exception:
                return default

    # session-state init
    if 'cfg' not in st.session_state:
        st.session_state.cfg = {
            'endpoint': safe_get_secret_streamlit('FACE_ENDPOINT', ''),
            'auth_method': 'Subscription Key',
            'key': safe_get_secret_streamlit('FACE_KEY', ''),
            'use_managed_identity': False,
            'timeout': 10,
            'retries': 3,
            'backoff_factor': 0.3,
            'recognition_model': DEFAULT_RECOGNITION_MODEL,
            'detection_model': DEFAULT_DETECTION_MODEL,
            # flag: habilitar features de identificación (solo si aprobado)
            'enable_identification_features': False
        }
    if 'log' not in st.session_state:
        st.session_state.log = []

    def log(msg, level='info'):
        st.session_state.log.insert(0, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {level.upper()}: {msg}")

    # small wrapper to call core do_request and capture errors
    def try_core(fn, *args, **kwargs):
        try:
            return fn(st.session_state.cfg, *args, **kwargs)
        except Exception as e:
            log(str(e), 'error')
            st.error(str(e))
            return None

    st.set_page_config(page_title='Azure Face API GUI (Attributes)', layout='wide')
    st.title('Azure Face API — Streamlit GUI (Attributes)')

    tabs = st.tabs(["Config", "Detect", "Attributes", "Verify", "Identify", "FindSimilar", "Collections", "Batch", "Liveness", "Logs"]) 

    # ---------- Config Tab ----------
    with tabs[0]:
        st.header('Configuración')
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader('Credenciales y endpoint')
            auth_method = st.selectbox('Método de autenticación', ['Subscription Key', 'Azure AD'], index=0 if st.session_state.cfg.get('auth_method','Subscription Key')=='Subscription Key' else 1, key='cfg_auth_method')
            st.session_state.cfg['auth_method'] = auth_method

            endpoint = st.text_input('Endpoint (ej. https://<mi-recurso>.cognitiveservices.azure.com)', value=st.session_state.cfg.get('endpoint',''), key='cfg_endpoint')
            if auth_method == 'Subscription Key':
                key = st.text_input('Subscription Key (Ocp-Apim-Subscription-Key)', value=st.session_state.cfg.get('key',''), type='password', key='cfg_key')
                st.session_state.cfg['key'] = key.strip()
                st.session_state.cfg['use_managed_identity'] = False
            else:
                use_mi = st.checkbox('Usar Managed Identity / DefaultAzureCredential', value=st.session_state.cfg.get('use_managed_identity', False), key='cfg_use_mi')
                st.session_state.cfg['use_managed_identity'] = use_mi
                st.write('La app intentará usar DefaultAzureCredential. En local añade variables de entorno AZURE_* o usa autenticación interactiva.')

            st.session_state.cfg['endpoint'] = endpoint.strip()

            st.markdown('**Parámetros de cliente (retries/timeout)**')
            col_a, col_b, col_c = st.columns(3)
            st.session_state.cfg['timeout'] = col_a.number_input('Timeout (s)', value=st.session_state.cfg.get('timeout',10), min_value=1, max_value=120, key='cfg_timeout')
            st.session_state.cfg['retries'] = col_b.number_input('Retries', value=st.session_state.cfg.get('retries',3), min_value=0, max_value=10, key='cfg_retries')
            st.session_state.cfg['backoff_factor'] = col_c.number_input('Backoff factor', value=st.session_state.cfg.get('backoff_factor',0.3), min_value=0.0, max_value=10.0, key='cfg_backoff')

            st.markdown('**Modelos**')
            st.session_state.cfg['recognition_model'] = st.selectbox('Recognition model', ["recognition_04","recognition_03","recognition_02","recognition_01"], index=0, key='cfg_recog_model')
            st.session_state.cfg['detection_model'] = st.selectbox('Detection model', ["detection_03","detection_02","detection_01"], index=0, key='cfg_det_model')

            # enable identification features flag
            st.session_state.cfg['enable_identification_features'] = st.checkbox('Habilitar IDENTIFICACIÓN/VERIFICACIÓN/persongroups (solo si Microsoft te aprobó)', value=st.session_state.cfg.get('enable_identification_features', False), key='cfg_enable_id')

            if st.button('Guardar configuración en sesión', key='cfg_save_btn'):
                st.success('Configuración guardada en sesión (no persistente).')
                log('Configuración actualizada por usuario')

            st.markdown('---')
            st.subheader('Guardar / cargar configuraciones')
            cfg_name = st.text_input('Nombre para guardar config', value='default', key='cfg_name')
            if st.button('Guardar config local', key='cfg_save_local'):
                save_config_to_db(cfg_name, st.session_state.cfg)
                st.success(f'Configuración guardada como "{cfg_name}"')

            saved = list_saved_configs()
            if saved:
                sel = st.selectbox('Configs guardadas', options=[''] + [r[0] for r in saved], key='cfg_saved_sel')
                if sel:
                    if st.button('Cargar config seleccionada', key='cfg_load_db'):
                        cfg_loaded = load_config_from_db(sel)
                        if cfg_loaded:
                            st.session_state.cfg.update(cfg_loaded)
                            _maybe_rerun()
                    if st.button('Eliminar config seleccionada', key='cfg_del_db'):
                        delete_config_from_db(sel)
                        _maybe_rerun()

        with col2:
            st.subheader('Import / Export')
            st.markdown('Cargar desde archivo .env (KEY/ENDPOINT) o config.json con las mismas keys.')
            uploaded = st.file_uploader('Subir .env o config.json', type=['env','txt','json'], key='cfg_upload')
            if uploaded and st.button('Cargar archivo', key='cfg_load_file'):
                data = uploaded.read()
                try:
                    if uploaded.name.endswith('.json'):
                        parsed = json.loads(data.decode('utf-8'))
                    else:
                        # simple parse .env
                        parsed = {}
                        for line in data.decode('utf-8').splitlines():
                            if '=' in line and not line.strip().startswith('#'):
                                k, v = line.split('=',1)
                                parsed[k.strip()] = v.strip().strip('"')
                    # map keys
                    if 'FACE_ENDPOINT' in parsed:
                        st.session_state.cfg['endpoint'] = parsed.get('FACE_ENDPOINT')
                    if 'FACE_KEY' in parsed:
                        st.session_state.cfg['key'] = parsed.get('FACE_KEY')
                    if 'AUTH_METHOD' in parsed:
                        st.session_state.cfg['auth_method'] = parsed.get('AUTH_METHOD')
                    st.success('Configuración cargada al estado de sesión (revisa y guarda).')
                    log('Configuración cargada desde archivo')
                except Exception as e:
                    st.error(f'Error cargando archivo: {e}')

            st.markdown('Descargar configuración actual')
            cfg_json = json.dumps(st.session_state.cfg, indent=2)
            st.download_button('Descargar config (JSON)', data=cfg_json, file_name='face_gui_config.json', mime='application/json', key='cfg_download')

            st.markdown('---')
            st.subheader('Probar conexión')
            if st.button('Test connection', key='cfg_test_conn'):
                try:
                    res = try_core(list_large_person_groups_core)
                    if res is not None:
                        st.success('Connection OK')
                except Exception as e:
                    st.error(str(e))

    # ---------- Detect Tab ----------
    with tabs[1]:
        st.header('Detectar rostros')
        uploaded = st.file_uploader('Sube imagen para detectar', type=['jpg','jpeg','png'], key='detect_upload')
        detect_landmarks = st.checkbox('Devolver landmarks', value=False, key='detect_landmarks')
        recognition_model = st.selectbox('Recognition model (detect)', ["recognition_04","recognition_03","recognition_02","recognition_01"], index=0, key='detect_recog')
        detection_model = st.selectbox('Detection model (detect)', ["detection_03","detection_02","detection_01"], index=0, key='detect_det')
        # decide whether to request faceId based on config flag
        allow_face_id = st.session_state.cfg.get('enable_identification_features', False)
        if allow_face_id:
            st.info('Has habilitado IDENTIFICACIÓN: la API intentará devolver faceId (solo si tu recurso está aprobado).')
        else:
            st.caption('Por defecto no se solicita faceId para evitar 403 UnsupportedFeature. Activa IDENTIFICACIÓN en Config si tu recurso ha sido aprobado.')
        if uploaded:
            img_bytes = uploaded.read()
            faces = try_core(detect_faces_core, img_bytes, return_landmarks=detect_landmarks, return_face_id=allow_face_id, recognition_model=recognition_model, detection_model=detection_model)
            if faces is not None:
                # draw boxes
                image = Image.open(BytesIO(img_bytes)).convert('RGB')
                draw = ImageDraw.Draw(image)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                for i, f in enumerate(faces):
                    r = f.get('faceRectangle', {})
                    left = r.get('left'); top = r.get('top'); width = r.get('width'); height = r.get('height')
                    if None in (left, top, width, height):
                        continue
                    box = [left, top, left + width, top + height]
                    draw.rectangle(box, outline='red', width=3)
                    draw.text((left, top - 10), f"#{i}", fill='white', font=font)
                st.image(image, use_column_width=True)
                st.json(faces)

    # ---------- Attributes Tab (nuevo) ----------
    with tabs[2]:
        st.header('Face Attributes — reconocimiento de características')
        st.markdown('Selecciona qué atributos quieres que Face API devuelva para la imagen subida. Algunos atributos requieren modelos/planes concretos.')
        attr_all = [
            'age','gender','headPose','smile','facialHair','glasses','emotion',
            'hair','makeup','occlusion','accessories','blur','exposure','noise'
        ]
        st.markdown('**Atributos disponibles**: ' + ', '.join(attr_all))
        uploaded_attr = st.file_uploader('Sube imagen para analizar atributos', type=['jpg','jpeg','png'], key='attr_upload')
        default_checked = ['age','gender','emotion','hair','glasses']
        cols = st.columns(3)
        selected_attrs = []
        for i, a in enumerate(attr_all):
            c = cols[i % 3].checkbox(a, value=(a in default_checked), key=f'attr_chk_{a}')
            if c:
                selected_attrs.append(a)
        if uploaded_attr and st.button('Analizar atributos', key='attr_analyze'):
            img = uploaded_attr.read()
            # build attributes param
            attrs_param = ','.join(selected_attrs) if selected_attrs else None
            faces = try_core(detect_faces_core, img, return_landmarks=True, return_face_id=allow_face_id, recognition_model=st.session_state.cfg.get('recognition_model'), detection_model=st.session_state.cfg.get('detection_model'), return_face_attributes=attrs_param)
            if not faces:
                st.warning('No se detectaron caras')
            else:
                # build a readable table of attributes
                rows = []
                for idx, f in enumerate(faces):
                    face_attr = f.get('faceAttributes', {})
                    row = {'faceIndex': idx, 'faceId': f.get('faceId')}
                    # basic attributes
                    row['age'] = face_attr.get('age')
                    row['gender'] = face_attr.get('gender')
                    row['smile'] = face_attr.get('smile')
                    # headPose
                    hp = face_attr.get('headPose')
                    if hp:
                        row['headPose_roll'] = hp.get('roll')
                        row['headPose_yaw'] = hp.get('yaw')
                        row['headPose_pitch'] = hp.get('pitch')
                    # facialHair
                    fh = face_attr.get('facialHair')
                    if fh:
                        row['facialHair_moustache'] = fh.get('moustache')
                        row['facialHair_beard'] = fh.get('beard')
                        row['facialHair_sideburns'] = fh.get('sideburns')
                    # glasses
                    row['glasses'] = face_attr.get('glasses')
                    # emotion: dominant
                    emo = face_attr.get('emotion') or {}
                    if emo:
                        dominant = max(emo.items(), key=lambda x: x[1]) if emo.items() else (None, None)
                        row['emotion_dominant'] = dominant[0]
                        row.update({f'emotion_{k}': v for k, v in emo.items()})
                    # hair
                    hair = face_attr.get('hair') or {}
                    row['hair_bald'] = hair.get('bald')
                    hair_colors = hair.get('hairColor') or []
                    if hair_colors:
                        # pick highest confidence
                        hc = sorted(hair_colors, key=lambda x: x.get('confidence',0), reverse=True)[0]
                        row['hair_color'] = hc.get('color')
                        row['hair_color_confidence'] = hc.get('confidence')
                    # makeup
                    makeup = face_attr.get('makeup') or {}
                    row['makeup_eyeMakeup'] = makeup.get('eyeMakeup')
                    row['makeup_lipMakeup'] = makeup.get('lipMakeup')
                    # occlusion
                    occ = face_attr.get('occlusion') or {}
                    row['occlusion_forehead'] = occ.get('foreheadOccluded')
                    row['occlusion_eyeLeft'] = occ.get('eyeLeftOccluded')
                    row['occlusion_eyeRight'] = occ.get('eyeRightOccluded')
                    # accessories
                    accessories = face_attr.get('accessories') or []
                    row['accessories'] = ','.join([a.get('type') for a in accessories]) if accessories else None
                    # image quality
                    blur = face_attr.get('blur') or {}
                    row['blur_blurLevel'] = blur.get('blurLevel')
                    exposure = face_attr.get('exposure') or {}
                    row['exposure_exposureLevel'] = exposure.get('exposureLevel')
                    noise = face_attr.get('noise') or {}
                    row['noise_noiseLevel'] = noise.get('noiseLevel')

                    rows.append(row)
                df = pd.DataFrame(rows)
                st.table(df)
                # also show full json per face
                for i, f in enumerate(faces):
                    st.subheader(f'Face #{i} JSON')
                    st.json(f)

    # ---------- Verify Tab ----------
    with tabs[3]:
        st.header('Verificar (one-to-one)')
        col_a, col_b = st.columns(2)
        img1 = col_a.file_uploader('Imagen A', key='v_img1', type=['jpg','jpeg','png'])
        img2 = col_b.file_uploader('Imagen B', key='v_img2', type=['jpg','jpeg','png'])
        if img1 and img2:
            b1 = img1.read(); b2 = img2.read()
            f1 = try_core(detect_faces_core, b1, return_landmarks=False, return_face_id=allow_face_id, recognition_model=st.session_state.cfg.get('recognition_model'), detection_model=st.session_state.cfg.get('detection_model'))
            f2 = try_core(detect_faces_core, b2, return_landmarks=False, return_face_id=allow_face_id, recognition_model=st.session_state.cfg.get('recognition_model'), detection_model=st.session_state.cfg.get('detection_model'))
            if f1 and f2 and len(f1)>0 and len(f2)>0:
                if not f1[0].get('faceId') or not f2[0].get('faceId'):
                    st.error('No se devolvió faceId en la detección. Activa IDENTIFICACIÓN en Config si tu recurso está aprobado por Microsoft.')
                else:
                    res = try_core(verify_faces_core, f1[0]['faceId'], f2[0]['faceId'])
                    if res:
                        st.write('¿Mismo sujeto? ->', res.get('isIdentical'))
                        st.write('Confianza ->', res.get('confidence'))
            else:
                st.warning('No se detectó cara en una de las imágenes')

    # ---------- Identify Tab ----------
    with tabs[4]:
        st.header('Identificar (one-to-many)')
        id_img = st.file_uploader('Imagen para identificar', type=['jpg','jpeg','png'], key='identify_img')
        groups = try_core(list_large_person_groups_core) or []
        group_options = {g['largePersonGroupId']: g.get('name') for g in groups} if groups else {}
        selected_group = st.selectbox('LargePersonGroup', options=[''] + list(group_options.keys()), key='identify_group')
        max_candidates = st.slider('Max candidatos', 1, 10, 5, key='identify_maxc')
        conf_thr = st.slider('Confidence threshold', 0.0, 1.0, 0.5, key='identify_conf')
        if id_img and selected_group:
            b = id_img.read()
            faces = try_core(detect_faces_core, b, return_landmarks=False, return_face_id=allow_face_id, recognition_model=st.session_state.cfg.get('recognition_model'), detection_model=st.session_state.cfg.get('detection_model'))
            if faces:
                # ensure faceIds exist
                faceIds = [f.get('faceId') for f in faces if f.get('faceId')]
                if not faceIds:
                    st.error('No se devolvieron faceId. Activa IDENTIFICACIÓN en Config si tu recurso está aprobado.')
                else:
                    res = try_core(identify_core, faceIds, selected_group, maxNumOfCandidatesReturned=max_candidates, confidenceThreshold=conf_thr)
                    if res:
                        st.json(res)
                        persons = try_core(list_persons_in_group_core, selected_group) or []
                        persons_map = {p['personId']: p for p in persons}
                        out = []
                        for r in res:
                            fid = r.get('faceId')
                            for c in r.get('candidates', []):
                                pid = c.get('personId')
                                out.append({'faceId': fid, 'personId': pid, 'personName': persons_map.get(pid, {}).get('name'), 'confidence': c.get('confidence')})
                        if out:
                            st.table(pd.DataFrame(out))

    # ---------- FindSimilar Tab ----------
    with tabs[5]:
        st.header('Find Similar')
        f_img = st.file_uploader('Imagen de consulta', type=['jpg','jpeg','png'], key='fs_img')
        large_face_list = st.text_input('LargeFaceListId (si aplica)', key='fs_lfl')
        maxcand = st.slider('Max candidatos', 1, 10, 5, key='fs_maxc')
        if f_img:
            b = f_img.read()
            faces = try_core(detect_faces_core, b, return_landmarks=False, return_face_id=allow_face_id, recognition_model=st.session_state.cfg.get('recognition_model'), detection_model=st.session_state.cfg.get('detection_model'))
            if faces:
                fid = faces[0].get('faceId')
                if not fid:
                    st.error('No se devolvió faceId. Activa IDENTIFICACIÓN en Config si tu recurso está aprobado.')
                else:
                    res = try_core(find_similar_core, fid, largeFaceListId=(large_face_list or None), maxNumOfCandidatesReturned=maxcand)
                    if res:
                        st.json(res)

    # ---------- Collections Tab (incluye A: persona detalle) ----------
    with tabs[6]:
        st.header('Collections / LargePersonGroup')
        st.subheader('Crear LargePersonGroup')
        col_a, col_b = st.columns(2)
        group_id = col_a.text_input('Group ID (slug)', key='col_group_id')
        group_name = col_b.text_input('Group name', key='col_group_name')
        if st.button('Crear grupo', key='col_create_group'):
            res = try_core(create_large_person_group_core, group_id.strip(), group_name.strip(), st.session_state.cfg.get('recognition_model'))
            if res is not None:
                st.success('Grupo creado')

        st.markdown('---')
        st.subheader('Crear persona y añadir caras')
        groups = try_core(list_large_person_groups_core) or []
        group_map = {g['largePersonGroupId']: g.get('name') for g in groups} if groups else {}
        sel_group = st.selectbox('Seleccionar grupo', options=[''] + list(group_map.keys()), key='col_sel_group')
        person_name = st.text_input('Nombre persona', key='col_person_name')
        uploaded_face = st.file_uploader('Subir imagen de la cara (1 por upload)', key='col_add_face')
        if st.button('Crear persona', key='col_create_person'):
            if not sel_group or not person_name:
                st.error('Selecciona un grupo y un nombre')
            else:
                res = try_core(create_person_core, sel_group, person_name)
                if res:
                    st.success(f"Persona creada: {res.get('personId')}")
        if uploaded_face and st.button('Añadir cara a persona', key='col_addface_btn'):
            person_id = st.text_input('PersonId (pegar aquí)', key='col_person_id_for_add')
            if not sel_group or not person_id:
                st.error('Selecciona grupo y pega el personId')
            else:
                data = uploaded_face.read()
                res = try_core(add_face_to_person_core, sel_group, person_id, data)
                if res:
                    st.success(f"persistedFaceId: {res.get('persistedFaceId')}")

        st.markdown('---')
        st.subheader('Entrenar grupo')
        sel_group2 = st.selectbox('Selecciona grupo a entrenar', options=[''] + list(group_map.keys()), key='col_train_group')
        if st.button('Iniciar entrenamiento', key='col_train_btn'):
            res = try_core(train_large_person_group_core, sel_group2)
            if res is not None:
                st.success('Entrenamiento lanzado')
        if st.button('Ver estado de entrenamiento', key='col_train_status'):
            status = try_core(get_training_status_core, sel_group2)
            if status:
                st.json(status)

        st.markdown('---')
        st.subheader('Listar persons (paginado)')
        sel_group3 = st.selectbox('Selecciona grupo para listar persons', options=[''] + list(group_map.keys()), key='col_list_group')
        page_size = st.number_input('Tamaño de página', value=20, min_value=1, max_value=100, key='col_page_size')
        page = st.number_input('Página (0-indexed)', value=0, min_value=0, key='col_page_index')
        if st.button('Listar persons en grupo', key='col_list_btn'):
            skip = int(page) * int(page_size)
            persons = try_core(list_persons_in_group_core, sel_group3, top=int(page_size), skip=skip)
            if persons is not None:
                st.write(f"{len(persons)} persons (page {page})")
                if persons:
                    df = pd.DataFrame(persons)
                    st.table(df)
                    csv = df.to_csv(index=False)
                    st.download_button('Exportar persons a CSV', data=csv, file_name=f'persons_{sel_group3}_page{page}.csv', key='col_export_csv')

        st.markdown('---')
        st.subheader('Detalle persona (A): ver persistedFaceIds / borrar persistedFace')
        detail_group = st.selectbox('Group for detail', options=[''] + list(group_map.keys()), key='detail_group')
        persons_for_detail = try_core(list_persons_in_group_core, detail_group) or []
        sel_person = st.selectbox('Seleccionar person', options=[''] + [p['personId'] for p in persons_for_detail], key='detail_person')
        if sel_person:
            person_detail = try_core(get_person_core, detail_group, sel_person)
            if person_detail:
                st.json(person_detail)
                persisted = person_detail.get('persistedFaceIds', [])
                st.write('persistedFaceIds:')
                for pf in persisted:
                    cols = st.columns([6,1,1])
                    cols[0].write(pf)
                    if cols[1].button('Borrar persistedFace', key=f'del_pf_{pf}'):
                        try:
                            delete_persisted_face_core(st.session_state.cfg, detail_group, sel_person, pf)
                            st.success('Persisted face borrado (actualiza listado)')
                            _maybe_rerun()
                        except Exception as e:
                            st.error(str(e))

    # ---------- Batch Tab ----------
    with tabs[7]:
        st.header('Batch: crear persons y subir caras desde ZIP')
        st.markdown('Sube un ZIP con imágenes y un CSV (name,image_filename) para crear persons y añadir caras. El CSV debe tener encabezado `name,image` o `name,image_filename`.')
        zip_file = st.file_uploader('ZIP (images + mapping.csv)', type=['zip'], key='batch_zip')
        target_group = st.text_input('LargePersonGroupId destino para el batch', key='batch_group')
        if zip_file and st.button('Procesar ZIP (batch)', key='batch_run'):
            if not target_group:
                st.error('Indica el largePersonGroupId destino')
            else:
                try:
                    z = zipfile.ZipFile(zip_file)
                    namelist = z.namelist()
                    csv_candidates = [n for n in namelist if n.lower().endswith('.csv')]
                    if not csv_candidates:
                        st.error('No se encontró CSV en el ZIP')
                    else:
                        csv_name = csv_candidates[0]
                        with z.open(csv_name) as fh:
                            txt = TextIOWrapper(fh, encoding='utf-8')
                            df = pd.read_csv(txt)
                        if 'name' not in df.columns or ('image' not in df.columns and 'image_filename' not in df.columns):
                            st.error('CSV debe contener columnas name y image (image_filename)')
                        else:
                            total = len(df)
                            progress = st.progress(0)
                            for i, row in df.iterrows():
                                person = row['name']
                                img_key = row.get('image') or row.get('image_filename')
                                if img_key not in namelist:
                                    st.warning(f"Imagen {img_key} no encontrada en ZIP, se omite")
                                    progress.progress(int((i+1)/total*100))
                                    continue
                                p = try_core(create_person_core, target_group, person)
                                pid = p.get('personId') if p else None
                                if pid:
                                    with z.open(img_key) as imgfh:
                                        b = imgfh.read()
                                        try_core(add_face_to_person_core, target_group, pid, b)
                                progress.progress(int((i+1)/total*100))
                            st.success('Batch procesado (revisa logs).')
                except Exception as e:
                    st.error(str(e))

    # ---------- Liveness Tab ----------
    with tabs[8]:
        st.header('Liveness (sesiones)')
        st.markdown("""Esta sección crea una sesión de liveness (server side) y permite consultar resultados. La parte de cliente (subir vídeo/selfie desde dispositivo para la sesión) debe integrarse en tu frontend usando el token/URL de la sesión.

    **Importante**: el servicio Liveness puede requerir acceso por separado y está sujeto a políticas de uso de Face. """)
        l_mode = st.selectbox('Modo', ['active','passive'], key='liveness_mode')
        verify_img = st.file_uploader('Opcional: imagen para verificar durante la sesión (verify image)', key='liveness_verify')
        if st.button('Crear sesión de liveness', key='liveness_create'):
            v = verify_img.read() if verify_img else None
            res = try_core(create_liveness_session_core, mode := l_mode, verify_image_bytes := v)
            if res:
                st.json(res)
        sess_id = st.text_input('SessionId para consultar', key='liveness_sess_id')
        if st.button('Consultar sesión', key='liveness_query'):
            res = try_core(get_liveness_session_core, sess_id)
            if res:
                st.json(res)

    # ---------- Logs Tab ----------
    with tabs[9]:
        st.header('Logs recientes')
        for line in st.session_state.log:
            st.text(line)

    st.markdown('---')
    st.caption('Recuerda: el uso de tecnologías biométricas tiene implicaciones legales y de privacidad. Usa solo con consentimiento y según la normativa aplicable.')

# Run streamlit UI only when executed as script (so uvicorn can import module without launching UI)
if __name__ == '__main__':
    run_streamlit()
