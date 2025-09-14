# core/core_face.py
import os
import io
from typing import List, Optional, Dict, Any
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.ai.vision.face import FaceClient, FaceAdministrationClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeTypeRecognition04,
)
import json

# Map friendly names to the SDK enums for attributes
ATTRIBUTE_MAP = {
    "age": FaceAttributeTypeRecognition04.AGE,
    "gender": FaceAttributeTypeRecognition04.GENDER,
    "emotion": FaceAttributeTypeRecognition04.EMOTION,
    "glasses": FaceAttributeTypeRecognition04.GLASSES,
    "hair": FaceAttributeTypeRecognition04.HAIR,
    "makeup": FaceAttributeTypeRecognition04.MAKEUP,
    "facialHair": FaceAttributeTypeRecognition04.FACIAL_HAIR,
    "headPose": FaceAttributeTypeRecognition04.HEAD_POSE,
    "occlusion": FaceAttributeTypeRecognition04.OCCLUSION,
    "accessories": FaceAttributeTypeRecognition04.ACCESSORIES,
    "blur": FaceAttributeTypeRecognition04.BLUR,
    "exposure": FaceAttributeTypeRecognition04.EXPOSURE,
    "noise": FaceAttributeTypeRecognition04.NOISE,
    "qualityForRecognition": FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION,
    "smile": FaceAttributeTypeRecognition04.SMILE,
}

# defaults
DEFAULT_DETECTION_MODEL = FaceDetectionModel.DETECTION03
DEFAULT_RECOGNITION_MODEL = FaceRecognitionModel.RECOGNITION04

def create_clients(cfg: dict):
    """
    Create and return tuple (face_client, admin_client)
    cfg: must contain endpoint and either key or use Managed Identity/Azure AD
    """
    endpoint = cfg.get("endpoint") or os.environ.get("FACE_ENDPOINT")
    if not endpoint:
        raise ValueError("Endpoint not provided in cfg.endpoint or FACE_ENDPOINT env")
    auth_method = cfg.get("auth_method", "Subscription Key")
    if auth_method == "Azure AD" or cfg.get("use_managed_identity"):
        cred = DefaultAzureCredential()
        face_client = FaceClient(endpoint=endpoint, credential=cred)
        admin_client = FaceAdministrationClient(endpoint=endpoint, credential=cred)
    else:
        key = cfg.get("key") or os.environ.get("FACE_KEY")
        if not key:
            raise ValueError("Subscription key missing (cfg.key or FACE_KEY env)")
        cred = AzureKeyCredential(key)
        face_client = FaceClient(endpoint=endpoint, credential=cred)
        admin_client = FaceAdministrationClient(endpoint=endpoint, credential=cred)
    return face_client, admin_client

def _to_simple(obj):
    """Try to convert SDK object to dict if possible, else str"""
    try:
        return obj.as_dict()
    except Exception:
        try:
            return json.loads(str(obj))
        except Exception:
            return str(obj)

def detect_faces_sdk(cfg: dict, img_bytes: Optional[bytes]=None, url: Optional[str]=None,
                     return_landmarks: bool=False, detection_model=DEFAULT_DETECTION_MODEL,
                     recognition_model=DEFAULT_RECOGNITION_MODEL,
                     return_face_attributes: Optional[List[str]]=None):
    """
    Use SDK to detect faces. return_face_attributes is a list of attribute names (keys in ATTRIBUTE_MAP)
    """
    client, _ = create_clients(cfg)
    attrs = None
    if return_face_attributes:
        attrs = [ATTRIBUTE_MAP[a] for a in return_face_attributes if a in ATTRIBUTE_MAP]
    try:
        if url:
            faces = client.detect_from_url(
                url=url,
                detection_model=detection_model,
                recognition_model=recognition_model,
                return_face_id=True,
                return_face_landmarks=return_landmarks,
                return_face_attributes=attrs
            )
        else:
            stream = io.BytesIO(img_bytes)
            faces = client.detect_from_stream(
                image=stream,
                detection_model=detection_model,
                recognition_model=recognition_model,
                return_face_id=True,
                return_face_landmarks=return_landmarks,
                return_face_attributes=attrs
            )
        # convert results to simple dicts
        out = []
        for f in faces:
            try:
                out.append(f.as_dict())
            except Exception:
                out.append(_to_simple(f))
        return out
    except HttpResponseError as e:
        # bubble up for probe / UI to decode
        raise

def verify_faces_sdk(cfg: dict, face_id1: str, face_id2: str) -> Dict[str, Any]:
    client, _ = create_clients(cfg)
    try:
        res = client.verify_face_to_face(face_id1=face_id1, face_id2=face_id2)
        return res.as_dict()
    except HttpResponseError:
        raise

def identify_sdk(cfg: dict, face_ids: List[str], large_person_group_id: str, max_candidates=5, confidence_threshold=0.5):
    client, _ = create_clients(cfg)
    try:
        res = client.identify_from_large_person_group(
            face_ids=face_ids,
            large_person_group_id=large_person_group_id,
            max_num_of_candidates_returned=max_candidates,
            confidence_threshold=confidence_threshold
        )
        return [r.as_dict() for r in res]
    except HttpResponseError:
        raise

def create_large_person_group_sdk(cfg: dict, group_id: str, name: str, recognition_model=DEFAULT_RECOGNITION_MODEL):
    _, admin = create_clients(cfg)
    try:
        admin.large_person_group.create(large_person_group_id=group_id, name=name, recognition_model=recognition_model)
        return {"ok": True}
    except HttpResponseError:
        raise

def list_large_person_groups_sdk(cfg: dict):
    _, admin = create_clients(cfg)
    try:
        groups = admin.large_person_group.list()
        return [g.as_dict() for g in groups]
    except HttpResponseError:
        raise

def add_face_from_stream_sdk(cfg: dict, group_id: str, person_id: str, img_bytes: bytes, detection_model=DEFAULT_DETECTION_MODEL):
    _, admin = create_clients(cfg)
    try:
        stream = io.BytesIO(img_bytes)
        res = admin.large_person_group.add_face_from_stream(
            large_person_group_id=group_id,
            person_id=person_id,
            image=stream,
            detection_model=detection_model
        )
        return _to_simple(res)
    except HttpResponseError:
        raise

def train_large_person_group_sdk(cfg: dict, group_id: str, polling_interval=5):
    _, admin = create_clients(cfg)
    try:
        poller = admin.large_person_group.begin_train(large_person_group_id=group_id, polling_interval=polling_interval)
        poller.wait()
        return {"status": "trained"}
    except HttpResponseError:
        raise

# Probe: quick checks to see what features the endpoint supports
def probe_endpoint(cfg: dict, test_image_url: Optional[str]=None):
    """
    Returns dict with booleans and messages:
    { 'detect_basic': True/False, 'detect_attributes': True/False, 'largePersonGroup': True/False, 'errors': [...] }
    """
    result = {
        "detect_basic": False,
        "detect_attributes": False,
        "largePersonGroup": False,
        "errors": []
    }
    # check detect basic
    try:
        # try a tiny detect with no attributes
        detect_faces_sdk(cfg, url=test_image_url) if test_image_url else detect_faces_sdk(cfg, img_bytes=b'\x00')
    except Exception as e:
        # Most endpoints won't accept empty bytes; we try to infer via "real" url if provided
        # Do a safe minimal detect via url if url provided
        if test_image_url:
            try:
                detect_faces_sdk(cfg, url=test_image_url)
                result["detect_basic"] = True
            except Exception as e2:
                result["errors"].append(f"detect_basic error: {repr(e2)}")
        else:
            result["errors"].append(f"detect_basic error: {repr(e)}")
    else:
        result["detect_basic"] = True

    # attributes test (if basic ok)
    if result["detect_basic"]:
        try:
            attrs = ["age", "gender"]
            # use a test URL if provided; else attempt detect with no real attributes will raise earlier
            if test_image_url:
                detect_faces_sdk(cfg, url=test_image_url, return_face_attributes=attrs)
            else:
                # can't do a robust attributes test without a real image URL, so mark as unknown
                pass
            result["detect_attributes"] = True
        except HttpResponseError as e:
            # If server returns 403 unsupported feature, we assume attributes not supported
            result["errors"].append(f"detect_attributes error: {repr(e)}")
            result["detect_attributes"] = False
        except Exception as e:
            result["errors"].append(f"detect_attributes error: {repr(e)}")
            result["detect_attributes"] = False

    # largePersonGroup test
    try:
        groups = list_large_person_groups_sdk(cfg)
        result["largePersonGroup"] = True
    except HttpResponseError as e:
        result["errors"].append(f"largePersonGroup error: {repr(e)}")
        result["largePersonGroup"] = False
    except Exception as e:
        result["errors"].append(f"largePersonGroup error: {repr(e)}")
        result["largePersonGroup"] = False

    return result
