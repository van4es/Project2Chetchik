# C:\obychenie2\ocr_display.py

import sys
import cv2
import easyocr
from ultralytics import YOLO

def read_display_from_image(image_path: str, model_path: str):
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
    results = model.predict(source=image_path, conf=0.15, imgsz=960, save=False)

    # 3) Инициализируем EasyOCR без allowlist
    reader = easyocr.Reader(['en'], gpu=False)

    # Загружаем изображение один раз
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: не удалось прочитать изображение '{image_path}'")
        return
    h, w = img.shape[:2]

    # 4) Поиск бокса с display
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
                    return

                # Масштаб 2×
                new_w = int((x2p - x1p) * 2)
                new_h = int((y2p - y1p) * 2)
                roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                # Перевод в серое и бинаризация Otsu
                gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Сохраняем для отладки
                cv2.imwrite("C:/obychenie2/results/roi_for_ocr.png", thresh)

                # 5) Запускаем OCR с allowlist цифр и точки
                ocr_results = reader.readtext(thresh, allowlist='0123456789.')

                if not ocr_results:
                    print("Дисплей найден, но OCR не смог распознать цифры.")
                else:
                    # Берём вариант с максимальной уверенностью
                    best = max(ocr_results, key=lambda x: x[2])
                    _, text, prob = best
                    print(f"Распознанный текст: '{text}' (достоверность {prob:.2f})")
                return

    print("Дисплей (класс 'display') не найден на изображении.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Ошибка: укажите путь к изображению.")
        print(r"  python ocr_display.py C:\obychenie2\test_images\1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    MODEL_PATH = "C:/obychenie2/runs/detect/local_yolov11/weights/best.pt"

    read_display_from_image(image_path, MODEL_PATH)
