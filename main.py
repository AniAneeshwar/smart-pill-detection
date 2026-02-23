import cv2
import easyocr
import pyttsx3
from transformers import pipeline
from difflib import get_close_matches, SequenceMatcher
import re
import time

# -----------------------------
# Knowledge Base
# -----------------------------
medicine_info = {
    "paracetamol": "Used to reduce fever and relieve mild to moderate pain.",
    "ibuprofen": "Nonsteroidal anti-inflammatory drug used for pain, fever, and inflammation.",
    "cetirizine": "Antihistamine used to relieve allergy symptoms like sneezing, itching, watery eyes.",
    "aspirin": "Used to reduce pain, fever, or inflammation, and sometimes to prevent heart attacks.",
    "metformin hydrochloride": "Used to control blood sugar levels in people with type 2 diabetes.",
    "amoxicillin": "Antibiotic used to treat a variety of bacterial infections."
}

# -----------------------------
# Initialize heavy objects ONCE
# -----------------------------
engine = pyttsx3.init()
reader = easyocr.Reader(['en'], gpu=False)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# -----------------------------
# Utility Functions
# -----------------------------
def speak(text):
    print("[TTS]:", text)
    engine.say(text)
    engine.runAndWait()

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def auto_correct(candidate, valid_names, threshold=0.7):
    candidate = clean_text(candidate)
    best_match = None
    best_score = 0

    for name in valid_names:
        score = similarity(candidate, name)
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= threshold:
        return best_match, best_score
    return candidate, best_score

# -----------------------------
# OCR Extraction
# -----------------------------
def extract_tablet_name(image_path):
    results = reader.readtext(image_path)

    candidates = []
    for bbox, text, conf in results:
        text = clean_text(text)
        if len(text) >= 4 and conf > 0.4:
            candidates.append((text, conf))

    if not candidates:
        return "", 0.0

    # Pick the text with highest OCR confidence
    best_text, best_conf = max(candidates, key=lambda x: x[1])
    print("OCR Candidates:", candidates)
    print("Selected OCR Text:", best_text, "| OCR Confidence:", best_conf)
    return best_text, best_conf

# -----------------------------
# Medicine Description
# -----------------------------
def describe_medicine(text):
    corrected, match_score = auto_correct(text, list(medicine_info.keys()))

    print("Corrected Tablet Name:", corrected, "| Match Score:", round(match_score, 2))

    if corrected in medicine_info:
        return corrected, medicine_info[corrected], True, match_score

    # Fallback to QA model
    question = f"What is the use of {corrected}?"
    context = f"{corrected} is a tablet. Explain its usage and effects."
    answer = qa_pipeline(question=question, context=context)

    return corrected, answer.get('answer', ""), False, match_score

# -----------------------------
# Camera Capture
# -----------------------------
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None

    speak("Camera opened. Press space to capture the tablet image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Tablet OCR - Press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            img_name = f"tablet_{int(time.time())}.jpg"
            cv2.imwrite(img_name, frame)
            print("Image captured:", img_name)
            cap.release()
            cv2.destroyAllWindows()
            return img_name

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

# -----------------------------
# Main Flow
# -----------------------------
def run_tablet_ocr_voice_system(max_retries=2):
    retries = 0

    while retries <= max_retries:
        image_path = capture_image()
        if image_path is None:
            speak("Camera error or cancelled. Exiting.")
            return

        tablet_name, ocr_conf = extract_tablet_name(image_path)
        if not tablet_name:
            speak("No readable tablet name found. Please try again.")
            retries += 1
            continue

        corrected_name, usage, known, match_score = describe_medicine(tablet_name)

        if known:
            speak(f"The tablet is {corrected_name}. {usage}")
            return

        if match_score < 0.6 or usage.strip() == "":
            speak("Tablet name is unclear. Please rescan the tablet image.")
            retries += 1
            continue

        # If unknown but model gave some answer
        speak(f"I am not fully sure. The tablet might be {corrected_name}. {usage}")
        return

    speak("Unable to identify the tablet after multiple attempts. Please seek medical assistance.")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    run_tablet_ocr_voice_system()