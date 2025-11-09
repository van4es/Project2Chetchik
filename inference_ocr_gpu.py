import sys
import cv2
import easyocr
from ultralytics import YOLO

def read_display_gpu(image_path: str, model_path: str):
    """
    1) Загружает YOLOv11-модель из model_path на GPU
    2) Детектирует все боксы, ищет класс "Display"
    3) Вырезает ROI, масштабирует, бинаризует
    4) Запускает EasyOCR (digits only)
    5) Печатает распознанный текст или ошибку
    """
    # 1) Загружаем модель (автоматически на GPU, т.к. device="cuda:0")
    model = YOLO(model_path)

    # 2) Предсказание (list[Results]) на GPU
    results = model.predict(source=image_path, conf=0.15, imgsz=960, device="cuda:0", save=False)

    # 3) Инициализируем EasyOCR (цифры + точка)
    reader = easyocr.Reader(['en'], gpu=True)

    # 4) Загружаем исходное изображение (OpenCV, CPU)
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: не удалось прочитать изображение '{image_path}'")
        return
    h, w = img.shape[:2]

    # 5) Проходим по всем боксам, ищем “Display”
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label.lower() == "display":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 6) Добавляем padding (10 px)
                pad = 10
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(w, x2 + pad)
                y2p = min(h, y2 + pad)

                # 7) Вырезаем ROI
                roi = img[y1p:y2p, x1p:x2p]
                if roi.size == 0:
                    print("ERROR: некорректные координаты ROI")
                    return

                # 8) Масштабируем ROI в 2 раза (для мелких цифр)
                new_w = int((x2p - x1p) * 2)
                new_h = int((y2p - y1p) * 2)
                roi_scaled = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                # 9) Конвертируем в серое + бинаризация Otsu
                gray = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 10) Сохраняем для отладки
                cv2.imwrite("C:/obychenie2/results/roi_for_ocr.png", thresh)

                # 11) Запускаем OCR с allowlist цифр и точки (GPU)
                ocr_results = reader.readtext(thresh, allowlist='0123456789.')

                if not ocr_results:
                    print("Дисплей найден, но OCR не смог распознать цифры.")
                else:
                    # Выбираем вариант с наибольшей уверенность
                    best = max(ocr_results, key=lambda x: x[2])  # (bbox, text, prob)
                    _, text, prob = best
                    print(f"Распознанный текст: '{text}' (достоверность {prob:.2f})")
                return

    print("Дисплей (класс 'Display') не найден на изображении.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Ошибка: укажите путь к изображению.")
        print(r"  python inference_ocr_gpu.py C:\obychenie2\test_images\1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    MODEL_PATH = "C:/obychenie2/runs/detect/local_yolov11s_gpu/weights/best.pt"

    read_display_gpu(image_path, MODEL_PATH)

