"""
License Plate Recognition ML Microservice
FastAPI + YOLOv8 + EasyOCR + OpenCV

Models are loaded ONCE at module level (global scope) to avoid
~2s per-request latency from repeated model initialization.
"""

import re
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import easyocr

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Plate Recognition ML Service")

# ── Global model loading ──────────────────────────────────────────────
# Load the specialized License Plate model
MODEL_PATH = "plate_model.pt"
print(f"INFO: Loading License Plate YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=False)


# ── Helpers ──────────────────────────────────────────────────────────
def run_ocr(ocr_reader, image):
    try:
        results = ocr_reader.readtext(
            image, paragraph=False, detail=1,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
    except TypeError:
        results = ocr_reader.readtext(image, paragraph=False, detail=1)

    if not results:
        return "", 0.0
    texts, confidences = [], []
    for detection in results:
        text = detection[1].strip()
        if text:
            texts.append(text)
            confidences.append(detection[2])
    combined_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return combined_text, avg_confidence

def post_process_plate(raw_text: str) -> tuple[str, bool]:
    text = raw_text.upper()
    for char in [" ", "-", ".", ",", ":", "/", "_"]:
        text = text.replace(char, "")

    # Strip noise — longest first to avoid partial replacement
    for noise in ["INDIA", "IND", "1ND", "lND", "INID", "BHT"]:
        text = text.replace(noise, "")

    if not text:
        return "", False

    # ── BH series: positions 0,1 are DIGITS (year like 21,22,23) ──
    bh_chars = list(text)
    DIGIT_FIXES  = {"O":"0","I":"1","Z":"2","S":"5","G":"6","T":"7"}
    LETTER_FIXES = {
    "0": "O",
    "1": "I",
    "8": "B",
    "5": "S",
    "2": "Z",
    "6": "G",
    "4": "A",   # ← ADD THIS — 4 is commonly misread for A
}
    for i, ch in enumerate(bh_chars):
        if i in (0, 1):         bh_chars[i] = DIGIT_FIXES.get(ch, ch)
        elif i in (2, 3):       bh_chars[i] = LETTER_FIXES.get(ch, ch)
        elif i in (4,5,6,7):    bh_chars[i] = DIGIT_FIXES.get(ch, ch)
        elif i in (8, 9):       bh_chars[i] = LETTER_FIXES.get(ch, ch)
    bh_match = re.search(r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}", "".join(bh_chars))
    if bh_match:
        return bh_match.group(), True

    # ── Standard plate: positions 0,1 are LETTERS (state code) ──
    std_chars = list(text)
    LETTER_FIXES2 = {"0":"O","1":"I","8":"B","5":"S","2":"Z","6":"G"}
    for i, ch in enumerate(std_chars):
        if i in (0, 1):         std_chars[i] = LETTER_FIXES2.get(ch, ch)
        elif i in (2, 3):       std_chars[i] = DIGIT_FIXES.get(ch, ch)
        elif i in (4, 5):       std_chars[i] = LETTER_FIXES2.get(ch, ch)
        elif i in (6,7,8,9):    std_chars[i] = DIGIT_FIXES.get(ch, ch)
    std_match = re.search(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}", "".join(std_chars))
    if std_match:
        return std_match.group(), True

    return text, False

# ── Endpoint ─────────────────────────────────────────────────────────
@app.post("/recognize")
async def recognize_plate(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Unreadable image"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

    # Step 2 — YOLO inference
    results = model(img, verbose=False)
    boxes = results[0].boxes
    names = model.names

    if boxes is None or len(boxes) == 0:
        return {
            "plate": None,
            "confidence": 0,
            "yolo_confidence": 0,
            "ocr_confidence": 0,
            "regex_matched": False,
            "error": "No plate detected in image",
        }

    # Step 3 — Pick best box
    confidences = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confidences))
    best_conf = float(confidences[best_idx])
    cls_name = names[int(boxes.cls[best_idx])]
    print(f"YOLO: Detected {cls_name} ({best_conf:.2f})")
    
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    # Step 4 — Crop with 20% padding to help OCR context
    # Change this one line in Step 4:
    h_img, w_img = img.shape[:2]   # was: h_img, w_img, _ = img.shape
    # Step 4 — Change width padding from 20% to 30%
    pw = int((x2 - x1) * 0.30)   # was 0.20
    ph = int((y2 - y1) * 0.20)   # keep height same
    x1_p, y1_p = max(0, x1 - pw), max(0, y1 - ph)
    x2_p, y2_p = min(w_img, x2 + pw), min(h_img, y2 + ph)
    
    plate_img = img[y1_p:y2_p, x1_p:x2_p]

    if plate_img.size == 0:
        return {"plate": None, "confidence": 0, "error": "Empty crop"}

    # Step 5 — Resize
    plate_img = cv2.resize(plate_img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)

    # Step 6 — Triple-pass OCR
    # Pass A: Raw
    t_a, c_a = run_ocr(reader, plate_img)

    # Pass B: CLAHE
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    t_b, c_b = run_ocr(reader, cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))

    # Pass C: Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(plate_img, -1, kernel)
    t_c, c_c = run_ocr(reader, sharpened)

    print(f"OCR results: A='{t_a}', B='{t_b}', C='{t_c}'")
    
    ocr_results = [(t_a, c_a), (t_b, c_b), (t_c, c_c)]
    valid_ocr = [r for r in ocr_results if r[0].strip()]
    
    if not valid_ocr:
        return {
            "plate": None,
            "confidence": 0,
            "yolo_confidence": round(best_conf, 4),
            "ocr_confidence": 0,
            "regex_matched": False,
            "error": "Plate region found but text unreadable",
        }

    # Pick best non-empty OCR result
    # AFTER (prefers whichever pass yields a regex match)
    plate_text, regex_matched, best_ocr_conf = "", False, 0.0

    for text, conf in sorted(valid_ocr, key=lambda x: x[1], reverse=True):
        candidate, matched = post_process_plate(text)
        if matched:
            plate_text, regex_matched, best_ocr_conf = candidate, True, conf
            break

    # Fallback: use highest-confidence result even without regex match
    if not plate_text:
        raw_best, best_ocr_conf = max(valid_ocr, key=lambda x: x[1])
        plate_text, regex_matched = post_process_plate(raw_best)


    return {
        "plate": plate_text,
        "confidence": round((best_conf + best_ocr_conf) / 2, 4),
        "yolo_confidence": round(best_conf, 4),
        "ocr_confidence": round(best_ocr_conf, 4),
        "regex_matched": regex_matched,
        "error": None,
    }
    