# ui/streamlit_app.py
import streamlit as st
from core import core_face, core_storage, db
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import base64

def _draw_boxes_and_show(image_bytes, faces):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, f in enumerate(faces):
        r = f.get("faceRectangle") or f.get("face_rectangle") or {}
        left = r.get("left"); top = r.get("top"); width = r.get("width"); height = r.get("height")
        if None in (left, top, width, height):
            continue
        box = [left, top, left + width, top + height]
        draw.rectangle(box, outline="red", width=3)
        draw.text((left, top - 10), f"#{i}", fill="white", font=font)
    st.image(image, use_column_width=True)

def run_streamlit():
    st.set_page_config(page_title="Azure Face GUI", layout="wide")
    st.title("Azure Face API — GUI (probe + detect + verify)")

    if "cfg" not in st.session_state:
        st.session_state.cfg = {
            "endpoint": st.secrets.get("FACE_ENDPOINT", "") if hasattr(st, "secrets") else "",
            "key": st.secrets.get("FACE_KEY", "") if hasattr(st, "secrets") else "",
            "auth_method": "Subscription Key",
            "use_managed_identity": False
        }
    if "probe" not in st.session_state:
        st.session_state.probe = {"ran": False, "result": None}
    if "log" not in st.session_state:
        st.session_state.log = []

    def log(msg):
        st.session_state.log.insert(0, msg)

    tabs = st.tabs(["Config", "Probe", "Detect", "Verify", "Attributes", "Storage", "Collections", "Logs"])

    # ===== Config tab =====
    with tabs[0]:
        st.header("Configuration")
        st.session_state.cfg["endpoint"] = st.text_input("FACE endpoint", value=st.session_state.cfg.get("endpoint",""), key="ui_ep")
        st.session_state.cfg["auth_method"] = st.selectbox("Auth method", ["Subscription Key", "Azure AD"], index=0 if st.session_state.cfg.get("auth_method","Subscription Key")=="Subscription Key" else 1, key="ui_auth")
        if st.session_state.cfg["auth_method"] == "Subscription Key":
            st.session_state.cfg["key"] = st.text_input("FACE key", value=st.session_state.cfg.get("key",""), type="password", key="ui_key")
            st.session_state.cfg["use_managed_identity"] = False
        else:
            st.session_state.cfg["use_managed_identity"] = st.checkbox("Use Managed Identity / DefaultAzureCredential", value=False, key="ui_mi")
        st.write("Models recommended: detection_03 and recognition_04 (these are the modern ones).")
        st.markdown("---")
        if st.button("Save config to session", key="cfg_save_btn"):
            db.save_config("default", st.session_state.cfg)
            st.success("Saved to DB (as 'default')")

    # ===== Probe tab =====
    with tabs[1]:
        st.header("Probe endpoint (quick checks)")
        test_url = st.text_input("Optional test image URL (helps test attributes)", key="probe_url")
        if st.button("Run probe", key="btn_probe"):
            st.session_state.probe["ran"] = True
            try:
                res = core_face.probe_endpoint(st.session_state.cfg, test_image_url=test_url if test_url else None)
                st.session_state.probe["result"] = res
                st.write(res)
                if not res.get("detect_basic", False):
                    st.warning("Detect básico falló: revisa endpoint/key.")
                else:
                    st.success("Detect básico OK")
                if not res.get("detect_attributes", False):
                    st.info("Attributes not supported on this endpoint (UI will disable Attributes options).")
                if not res.get("largePersonGroup", False):
                    st.info("LargePersonGroup (Collections) not supported on this endpoint.")
            except Exception as e:
                st.error(f"Probe error: {e}")
                st.session_state.probe["result"] = {"errors":[str(e)]}

        if st.session_state.probe["ran"] and st.session_state.probe["result"]:
            st.json(st.session_state.probe["result"])

    # ===== Detect tab =====
    with tabs[2]:
        st.header("Detect faces")
        input_mode = st.radio("Input", ["Upload image", "Image URL"], index=0, key="detect_input_mode")
        uploaded = None
        img_url = None
        if input_mode == "Upload image":
            uploaded = st.file_uploader("Image (jpg/png)", type=["jpg","jpeg","png"], key="detect_upload")
        else:
            img_url = st.text_input("Image URL", key="detect_url")

        # show attributes options only if probe says ok (or unknown)
        probe_res = st.session_state.probe.get("result") or {}
        attrs_supported = probe_res.get("detect_attributes", None)
        if attrs_supported is False:
            st.info("This endpoint does not support face attributes (probe returned unsupported).")
        attr_choices = ["age","gender","emotion","glasses","hair","makeup","facialHair","headPose","occlusion","accessories","blur","exposure","noise","qualityForRecognition","smile"]
        selected_attrs = st.multiselect("Attributes to request (if supported)", options=attr_choices, default=["age","gender","emotion"], key="detect_attrs")

        detection_model = st.selectbox("Detection model", ["detection_03"], index=0, key="det_model_select")  # only modern option
        recognition_model = st.selectbox("Recognition model", ["recognition_04"], index=0, key="rec_model_select")  # only modern option

        if (uploaded or img_url) and st.button("Detect", key="btn_detect"):
            try:
                if uploaded:
                    img_bytes = uploaded.read()
                    faces = core_face.detect_faces_sdk(st.session_state.cfg, img_bytes=img_bytes, return_landmarks=True, return_face_attributes=selected_attrs)
                    if faces:
                        _draw_boxes_and_show(img_bytes, faces)
                        st.json(faces)
                    else:
                        st.warning("No faces detected.")
                else:
                    faces = core_face.detect_faces_sdk(st.session_state.cfg, url=img_url, return_landmarks=True, return_face_attributes=selected_attrs)
                    # download image for drawing
                    import requests
                    r = requests.get(img_url, timeout=10)
                    if r.status_code == 200:
                        _draw_boxes_and_show(r.content, faces)
                    st.json(faces)
            except Exception as e:
                st.error(f"Detect error: {e}")

    # ===== Verify tab =====
    with tabs[3]:
        st.header("Verify (one-to-one and one-to-group)")
        mode = st.radio("Mode", ["Face-to-Face", "Face-to-Group"], key="verify_mode")
        if mode == "Face-to-Face":
            col1, col2 = st.columns(2)
            with col1:
                img1 = st.file_uploader("Image A", type=["jpg","jpeg","png"], key="v_img1")
            with col2:
                img2 = st.file_uploader("Image B", type=["jpg","jpeg","png"], key="v_img2")
            if img1 and img2 and st.button("Verify faces (A vs B)", key="btn_verify_ff"):
                try:
                    b1 = img1.read(); b2 = img2.read()
                    f1 = core_face.detect_faces_sdk(st.session_state.cfg, img_bytes=b1, return_landmarks=False)
                    f2 = core_face.detect_faces_sdk(st.session_state.cfg, img_bytes=b2, return_landmarks=False)
                    if f1 and f2:
                        res = core_face.verify_faces_sdk(st.session_state.cfg, f1[0]["faceId"], f2[0]["faceId"])
                        st.json(res)
                    else:
                        st.warning("Could not detect faces in one of the images.")
                except Exception as e:
                    st.error(f"Verify error: {e}")
        else:
            st.write("Verify a detected face against a LargePersonGroup (requires group support)")
            group_id = st.text_input("LargePersonGroup ID", key="v_group")
            img = st.file_uploader("Image to verify", type=["jpg","jpeg","png"], key="v_img_group")
            if img and group_id and st.button("Verify against group", key="btn_verify_group"):
                try:
                    b = img.read()
                    faces = core_face.detect_faces_sdk(st.session_state.cfg, img_bytes=b, return_landmarks=False)
                    if not faces:
                        st.warning("No face detected")
                    else:
                        face_id = faces[0]["faceId"]
                        # identify
                        results = core_face.identify_sdk(st.session_state.cfg, [face_id], large_person_group_id=group_id)
                        st.json(results)
                except Exception as e:
                    st.error(f"Verify group error: {e}")

    # ===== Attributes tab (readable table) =====
    with tabs[4]:
        st.header("Attributes (readable table)")
        uploaded = st.file_uploader("Image (for attributes)", type=["jpg","jpeg","png"], key="attr_img")
        attrs = st.multiselect("Attributes", options=["age","gender","emotion","hair","glasses","facialHair","makeup","accessories","occlusion","blur","exposure","noise","qualityForRecognition","smile"], default=["age","gender","emotion"], key="attr_select")
        if uploaded and st.button("Analyze attributes", key="btn_attr_analyze"):
            try:
                b = uploaded.read()
                faces = core_face.detect_faces_sdk(st.session_state.cfg, img_bytes=b, return_landmarks=False, return_face_attributes=attrs)
                if not faces:
                    st.warning("No faces detected")
                else:
                    # build table
                    rows = []
                    for i, f in enumerate(faces):
                        fa = f.get("faceAttributes", {})
                        row = {"faceIndex": i, "faceId": f.get("faceId")}
                        row["age"] = fa.get("age")
                        row["gender"] = fa.get("gender")
                        emo = fa.get("emotion") or {}
                        if emo:
                            dom = max(emo.items(), key=lambda x: x[1])[0]
                            row["emotion_dominant"] = dom
                        hair = fa.get("hair") or {}
                        hc = (hair.get("hairColor") or [])
                        if hc:
                            row["hair_color"] = sorted(hc, key=lambda x: x.get("confidence",0), reverse=True)[0].get("color")
                        row["glasses"] = fa.get("glasses")
                        rows.append(row)
                    st.table(pd.DataFrame(rows))
                    st.json(faces)
            except Exception as e:
                st.error(f"Attributes error: {e}")

    # ===== Storage tab =====
    with tabs[5]:
        st.header("Storage (Blob / ADLS)")
        uploaded = st.file_uploader("Image to upload", type=["jpg","jpeg","png"], key="st_upload")
        container = st.text_input("Container / filesystem", value="face-images", key="st_container")
        path = st.text_input("Blob path (folder/name.jpg)", key="st_blob_path")
        if uploaded and st.button("Upload to blob", key="btn_upload_blob"):
            try:
                b = uploaded.read()
                cfg_storage = {
                    "blob_connection_string": st.secrets.get("BLOB_CONNECTION_STRING", "") if hasattr(st, "secrets") else "",
                    "blob_account_url": st.secrets.get("BLOB_ACCOUNT_URL", "")
                }
                url = core_storage.upload_blob_from_bytes(cfg_storage, container, path, b, overwrite=True)
                st.success("Uploaded: " + url)
            except Exception as e:
                st.error(f"Upload error: {e}")

    # ===== Collections (Groups) tab =====
    with tabs[6]:
        st.header("Collections (LargePersonGroup)")
        st.write("Create group / create person / add faces / train. These operations require the endpoint to support largePersonGroup.")
        gid = st.text_input("Group id", key="col_gid")
        gname = st.text_input("Group name", key="col_gname")
        if st.button("Create group", key="btn_create_group"):
            try:
                res = core_face.create_large_person_group_sdk(st.session_state.cfg, gid, gname)
                st.success("Group created")
            except Exception as e:
                st.error(f"Create group error: {e}")
        st.markdown("---")
        st.subheader("Add person and face")
        pg = st.text_input("Select group id", key="col_sel_gid")
        pname = st.text_input("Person name", key="col_person_name")
        face_file = st.file_uploader("Image for person", type=["jpg","jpeg","png"], key="col_face_file")
        if st.button("Create person and add face", key="btn_create_person"):
            try:
                # create person
                from core.core_face import create_clients
                _, admin = create_clients(st.session_state.cfg)
                person = admin.large_person_group.create_person(large_person_group_id=pg, name=pname)
                pid = person.person_id
                if face_file:
                    b = face_file.read()
                    admin.large_person_group.add_face_from_stream(large_person_group_id=pg, person_id=pid, image=BytesIO(b))
                st.success(f"Person created: {pid}")
            except Exception as e:
                st.error(f"Create person error: {e}")

    # ===== Logs =====
    with tabs[7]:
        st.header("Logs")
        for l in st.session_state.log:
            st.write(l)
