"""
face_engine.py
==============
แกนหลักของระบบตรวจจับและจดจำใบหน้า (Person/Face Detection + Whitelist Matching)

ใช้ OpenCV Haar Cascade สำหรับตรวจจับใบหน้า (Face Detection)
และ LBPH (Local Binary Patterns Histograms) สำหรับจดจำใบหน้า (Face Recognition)
- เป็นวิธีที่เบา ไม่ต้องใช้ GPU ไม่ต้องดาวน์โหลดโมเดลขนาดใหญ่ เหมาะกับ MVP/Prototype

หมายเหตุด้านความเป็นส่วนตัว:
- ภาพใบหน้าทั้งหมดถูกเก็บไว้ในเครื่อง (local) เท่านั้น ไม่มีการส่งออกไปที่ใดภายนอก
- ระบบนี้ใช้เพื่อสาธิตแนวคิดเท่านั้น การใช้งานจริงต้องขอความยินยอมจากผู้พักอาศัยก่อนเก็บภาพใบหน้า
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime

FACE_SIZE = (200, 200)

# ใช้ path แบบ absolute โดยอ้างอิงจากตำแหน่งไฟล์นี้เอง (ไม่พึ่ง cv2.data.haarcascades)
# เพราะบางเครื่อง Windows หาไฟล์จาก cv2.data ไม่เจอ (เช่น path มีอักษรไทย, การติดตั้งไม่สมบูรณ์)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(_BASE_DIR, "assets", "haarcascade_frontalface_default.xml")

WHITELIST_DIR = os.path.join(_BASE_DIR, "data", "whitelist")
MODEL_DIR = os.path.join(_BASE_DIR, "data", "model")
LOG_DIR = os.path.join(_BASE_DIR, "data", "logs")
MODEL_FILE = os.path.join(MODEL_DIR, "lbph_model.yml")
LABELS_FILE = os.path.join(MODEL_DIR, "labels.json")

os.makedirs(WHITELIST_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(
        f"ไม่พบไฟล์ cascade ที่ {CASCADE_PATH}\n"
        "กรุณาตรวจสอบว่าโฟลเดอร์ 'assets' อยู่ในตำแหน่งเดียวกับ face_engine.py"
    )

_detector = cv2.CascadeClassifier(CASCADE_PATH)
if _detector.empty():
    raise RuntimeError(
        f"โหลดไฟล์ cascade จาก {CASCADE_PATH} ไม่สำเร็จ (ไฟล์อาจเสียหายหรือไม่สมบูรณ์)\n"
        "ลองดาวน์โหลดโปรเจกต์ใหม่อีกครั้ง หรือแจ้งผู้พัฒนา"
    )


def detect_faces(gray_image):
    """ตรวจจับใบหน้าทั้งหมดในภาพขาวดำ คืนค่าเป็นลิสต์ของ (x, y, w, h)"""
    faces = _detector.detectMultiScale(
        gray_image, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60)
    )
    return faces


def _crop_and_normalize(gray_image, box):
    x, y, w, h = box
    face = gray_image[y : y + h, x : x + w]
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.equalizeHist(face)  # ปรับแสงให้สม่ำเสมอ ลดผลจากแสงน้อย/มาก
    return face


def list_whitelist_people():
    """คืนรายชื่อคนที่ลงทะเบียนไว้แล้ว พร้อมจำนวนรูปของแต่ละคน"""
    people = {}
    if not os.path.isdir(WHITELIST_DIR):
        return people
    for name in sorted(os.listdir(WHITELIST_DIR)):
        person_dir = os.path.join(WHITELIST_DIR, name)
        if os.path.isdir(person_dir):
            imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            people[name] = len(imgs)
    return people


def _imwrite_unicode(path, image):
    """
    บันทึกภาพลง path ที่อาจมีอักษรไทย/Unicode ได้อย่างถูกต้อง

    หมายเหตุ: cv2.imwrite() ธรรมดามีบั๊กบน Windows คือเขียนไฟล์ไปยัง path
    ที่มีอักษรที่ไม่ใช่ ASCII (เช่น ภาษาไทย) ไม่ได้ และจะคืนค่า False แบบเงียบ ๆ
    (ไม่ throw exception) ทำให้โปรแกรมเข้าใจผิดว่าบันทึกสำเร็จ
    วิธีแก้คือ encode ภาพเป็น bytes ด้วย cv2.imencode() ก่อน แล้วใช้ open() ของ
    Python เขียนไฟล์เอง เพราะ Python จัดการ Unicode path ได้ถูกต้องอยู่แล้ว
    """
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        with open(path, "wb") as f:
            f.write(encoded.tobytes())
        return True
    except OSError:
        return False


def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """อ่านภาพจาก path ที่อาจมีอักษรไทย/Unicode ได้อย่างถูกต้อง (คู่กับ _imwrite_unicode)"""
    try:
        with open(path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    except OSError:
        return None
    if file_bytes.size == 0:
        return None
    return cv2.imdecode(file_bytes, flags)


def register_face(name, image_bgr):
    """
    บันทึกภาพใบหน้าใหม่เข้า whitelist ของบุคคลชื่อ `name`
    คืนค่า True/False พร้อมข้อความ ว่าพบใบหน้าในภาพหรือไม่
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        return False, "ไม่พบใบหน้าในภาพนี้ กรุณาลองภาพอื่นที่เห็นใบหน้าชัดเจน"

    # ใช้ใบหน้าที่ใหญ่ที่สุดในภาพ (สมมติว่าเป็นตัวแบบหลัก)
    box = max(faces, key=lambda b: b[2] * b[3])
    face_crop = _crop_and_normalize(gray, box)

    person_dir = os.path.join(WHITELIST_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    existing = len([f for f in os.listdir(person_dir) if f.endswith(".png")])
    out_path = os.path.join(person_dir, f"{existing + 1:03d}.png")

    saved = _imwrite_unicode(out_path, face_crop)
    if not saved:
        return False, "บันทึกไฟล์ไม่สำเร็จ (เขียนไฟล์ลงดิสก์ไม่ได้) กรุณาลองใหม่อีกครั้ง"
    return True, f"บันทึกใบหน้าของ '{name}' สำเร็จ (รูปที่ {existing + 1})"


def delete_person(name):
    person_dir = os.path.join(WHITELIST_DIR, name)
    if os.path.isdir(person_dir):
        for f in os.listdir(person_dir):
            os.remove(os.path.join(person_dir, f))
        os.rmdir(person_dir)
        return True
    return False


def train_model():
    """
    เทรนโมเดล LBPH จากภาพทั้งหมดใน whitelist
    คืนค่า (สำเร็จหรือไม่, ข้อความ, จำนวนคน, จำนวนภาพทั้งหมด)
    """
    faces = []
    labels = []
    label_map = {}  # {label_id: name}
    next_label = 0

    if not os.path.isdir(WHITELIST_DIR):
        return False, "ยังไม่มีข้อมูล whitelist", 0, 0

    for name in sorted(os.listdir(WHITELIST_DIR)):
        person_dir = os.path.join(WHITELIST_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        img_files = [f for f in os.listdir(person_dir) if f.endswith(".png")]
        if not img_files:
            continue
        label_map[next_label] = name
        for f in img_files:
            img = _imread_unicode(os.path.join(person_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(next_label)
        next_label += 1

    if len(faces) == 0 or len(label_map) == 0:
        return False, "ยังไม่มีภาพใบหน้าที่ลงทะเบียนไว้ กรุณาเพิ่มข้อมูลก่อน", 0, 0

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    return True, f"เทรนโมเดลสำเร็จ: {len(label_map)} คน, {len(faces)} ภาพ", len(label_map), len(faces)


def _load_model():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE)):
        return None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    label_map = {int(k): v for k, v in raw_map.items()}
    return recognizer, label_map


def analyze_image(image_bgr, confidence_threshold=75):
    """
    วิเคราะห์ภาพหนึ่งภาพ: ตรวจจับใบหน้าทั้งหมด + จดจำว่าตรงกับ whitelist หรือไม่

    LBPH confidence: ค่ายิ่งต่ำ = ยิ่งมั่นใจว่าตรงกับคนนั้น (คนละหน่วยกับ % ความมั่นใจทั่วไป)
    ค่า threshold ยิ่งต่ำ = เข้มงวดมาก (โอกาส False Reject สูงขึ้น)
    ค่า threshold ยิ่งสูง = หลวมขึ้น (โอกาส False Accept สูงขึ้น)

    คืนค่า: annotated_image (BGR), results (list of dict), has_unknown (bool)
    """
    recognizer, label_map = _load_model()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    annotated = image_bgr.copy()
    results = []
    has_unknown = False

    for box in faces:
        x, y, w, h = box
        face_crop = _crop_and_normalize(gray, box)

        if recognizer is not None:
            label_id, confidence = recognizer.predict(face_crop)
            if confidence <= confidence_threshold:
                name = label_map.get(label_id, "ไม่ทราบชื่อ")
                status = "known"
            else:
                name = None
                status = "unknown"
        else:
            # ยังไม่มีโมเดล (ยังไม่เคยเทรน) -> ถือว่าทุกใบหน้าเป็นคนแปลกหน้า
            name = None
            confidence = None
            status = "unknown"

        if status == "unknown":
            has_unknown = True
            color = (0, 0, 255)  # แดง (BGR)
            label_text = "ไม่รู้จัก!"
        else:
            color = (0, 180, 0)  # เขียว
            label_text = name

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        # วาดพื้นหลังของป้ายชื่อให้อ่านง่าย (ไม่พึ่งฟอนต์ไทยใน OpenCV เพราะไม่รองรับ จึงใช้ label แบบย่อ)
        tag = "OK" if status == "known" else "ALERT"
        cv2.rectangle(annotated, (x, y - 28), (x + max(60, w), y), color, -1)
        cv2.putText(annotated, tag, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        results.append(
            {
                "box": [int(x), int(y), int(w), int(h)],
                "status": status,
                "name": name,
                "confidence": float(confidence) if confidence is not None else None,
            }
        )

    return annotated, results, has_unknown


def save_event_snapshot(annotated_image_bgr):
    """บันทึกภาพเหตุการณ์ พร้อม timestamp คืน path ของไฟล์ที่บันทึก"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(LOG_DIR, f"event_{ts}.jpg")
    _imwrite_unicode(path, annotated_image_bgr)
    return path
