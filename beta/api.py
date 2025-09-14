# api.py
"""
API wrapper for Azure Face API services.
Centralizes all calls to FaceClient and FaceAdministrationClient.
"""

import os
import uuid
from typing import List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceAdministrationClient, FaceClient
from azure.ai.vision.face.models import (
    FaceAttributeTypeRecognition04,
    FaceDetectionModel,
    FaceRecognitionModel,
    QualityForRecognition,
)

# Load from environment (can also be passed from Streamlit sidebar inputs)
KEY = os.getenv("FACE_APIKEY")
ENDPOINT = os.getenv("FACE_ENDPOINT")

# Create clients
face_admin_client = FaceAdministrationClient(
    endpoint=ENDPOINT, credential=AzureKeyCredential(KEY)
)
face_client = FaceClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))


# --------------------------
# Detection
# --------------------------
def detect_faces_from_file(
    file_path: str,
    detection_model: str = FaceDetectionModel.DETECTION03,
    recognition_model: str = FaceRecognitionModel.RECOGNITION04,
    return_attributes: Optional[List[str]] = None,
):
    """Detect faces in a local image file."""
    with open(file_path, "rb") as f:
        detected_faces = face_client.detect_from_stream(
            image=f,
            detection_model=detection_model,
            recognition_model=recognition_model,
            return_face_id=True,
            return_face_landmarks=True,
            return_face_attributes=return_attributes
            or [FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION],
        )
    return detected_faces


def detect_faces_from_url(
    url: str,
    detection_model: str = FaceDetectionModel.DETECTION03,
    recognition_model: str = FaceRecognitionModel.RECOGNITION04,
    return_attributes: Optional[List[str]] = None,
):
    """Detect faces in an image from URL."""
    detected_faces = face_client.detect_from_url(
        url=url,
        detection_model=detection_model,
        recognition_model=recognition_model,
        return_face_id=True,
        return_face_landmarks=True,
        return_face_attributes=return_attributes
        or [FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION],
    )
    return detected_faces


# --------------------------
# Verify
# --------------------------
def verify_faces(face_id1: str, face_id2: str):
    """Verify whether two face IDs belong to the same person."""
    return face_client.verify_face_to_face(face_id1=face_id1, face_id2=face_id2)


def verify_face_in_person_group(face_id: str, person_group_id: str, person_id: str):
    """Verify a detected face against a specific person in a group."""
    return face_client.verify_from_large_person_group(
        face_id=face_id, large_person_group_id=person_group_id, person_id=person_id
    )


# --------------------------
# Person Group Management
# --------------------------
def create_large_person_group(
    name: str = None,
    recognition_model: str = FaceRecognitionModel.RECOGNITION04,
) -> str:
    """Create a new Large Person Group with a random ID."""
    group_id = str(uuid.uuid4())
    face_admin_client.large_person_group.create(
        large_person_group_id=group_id,
        name=name or group_id,
        recognition_model=recognition_model,
    )
    return group_id


def add_person_to_group(group_id: str, person_name: str):
    """Create a new person inside a person group."""
    return face_admin_client.large_person_group.create_person(
        large_person_group_id=group_id, name=person_name
    )


def add_face_to_person_from_url(group_id: str, person_id: str, url: str):
    """Add a face image to an existing person in the group."""
    return face_admin_client.large_person_group.add_face_from_url(
        large_person_group_id=group_id,
        person_id=person_id,
        url=url,
        detection_model=FaceDetectionModel.DETECTION03,
    )


def train_person_group(group_id: str):
    """Train a large person group."""
    poller = face_admin_client.large_person_group.begin_train(
        large_person_group_id=group_id, polling_interval=5
    )
    poller.wait()
    return True


def identify_faces(group_id: str, face_ids: List[str]):
    """Identify faces against a trained Large Person Group."""
    return face_client.identify_from_large_person_group(
        face_ids=face_ids, large_person_group_id=group_id
    )


def delete_person_group(group_id: str):
    """Delete a Large Person Group."""
    face_admin_client.large_person_group.delete(group_id)
    return True
