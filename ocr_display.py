# C:\obychenie2\ocr_display.py

import sys
from pathlib import Path

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

def _preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """Подготовка вырезанного дисплея к OCR.

    Пайплайн концентрируется на цифрах: повышаем контраст, подавляем шум
    и получаем стабильную бинарную картинку. Возвращаем чёрно-белое изображение,
    которое отлично подходит для EasyOCR.
    """

    # Контраст и подавление шума
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)

    # Локальное выравнивание освещённости
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    # Адаптивная бинаризация + небольшая морфология для очистки цифр
    binary = cv2.adaptiveThreshold(
        enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=5,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return opened


def read_display_from_image(image_path: str, model_path: str, debug_dir: str | None = None):
    """
    1) Загружает YOLO‑модель из model_path
    2) Находит все боксы и ищет класс "display"
    3) Вырезает ROI, усиливает контраст/масштаб
    4) Запускает EasyOCR только с цифрами (allowlist)
    5) Печатает результат или сообщение об ошибке
    """
    # 1) Загружаем YOLO‑модель
    model = YOLO(model_path)

    # 2) Детекция (list[Results])
    results = model.predict(source=image_path, conf=0.2, imgsz=960, save=False)

    # 3) Инициализируем EasyOCR с кэшем
    reader = easyocr.Reader(['en'], gpu=False)

    # Загружаем изображение один раз
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: не удалось прочитать изображение '{image_path}'")
        return
    h, w = img.shape[:2]

    # 4) Поиск бокса с display
    best_candidate = None
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label.lower() == "display":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Добавим padding
                pad = 10
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(w, x2 + pad)
                y2p = min(h, y2 + pad)

                roi = img[y1p:y2p, x1p:x2p]
                if roi.size == 0:
                    print("ERROR: некорректные координаты дисплея")
                    continue

                # Масштаб 2×
                new_w = int((x2p - x1p) * 2)
                new_h = int((y2p - y1p) * 2)
                roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                processed = _preprocess_roi(roi_resized)

                if debug_dir:
                    dbg_path = Path(debug_dir)
                    dbg_path.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dbg_path / "roi_raw.png"), roi_resized)
                    cv2.imwrite(str(dbg_path / "roi_preprocessed.png"), processed)

                # 5) Запускаем OCR с allowlist цифр и спецсимволов
                ocr_results = reader.readtext(
                    processed,
                    allowlist="0123456789.-",
                    detail=1,
                    paragraph=False,
                )

                if not ocr_results:
                    continue

                best = max(ocr_results, key=lambda x: x[2])
                _, text, prob = best

                # Отбрасываем всё, что не похоже на число
                filtered = "".join(ch for ch in text if ch in "0123456789.-")
                if not filtered:
                    continue

                candidate = {
                    "text": filtered,
                    "prob": prob,
                    "area": (x2p - x1p) * (y2p - y1p),
                }

                if (
                    best_candidate is None
                    or candidate["prob"] > best_candidate["prob"] + 0.05
                    or (
                        abs(candidate["prob"] - best_candidate["prob"]) < 0.05
                        and candidate["area"] > best_candidate["area"]
                    )
                ):
                    best_candidate = candidate

    if best_candidate:
        print(
            f"Распознанный текст: '{best_candidate['text']}' "
            f"(достоверность {best_candidate['prob']:.2f})"
        )
        return

    print("Дисплей (класс 'display') не найден на изображении.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Ошибка: укажите путь к изображению.")
        print(r"  python ocr_display.py C:\obychenie2\test_images\1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    MODEL_PATH = "C:/obychenie2/runs/detect/local_yolov11/weights/best.pt"

    debug_folder = None
    if len(sys.argv) >= 3:
        debug_folder = sys.argv[2]

    read_display_from_image(image_path, MODEL_PATH, debug_folder)
