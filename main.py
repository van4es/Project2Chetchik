# C:\obychenie2\read_display.py

import sys
import cv2
import easyocr
from ultralytics import YOLO

def read_display_from_image(image_path: str, model_path: str):
    """
    1) Загружает модель YOLOv11 из model_path
    2) Делает детекцию объектов на image_path
    3) Ищет класс "display" (регистр буквы не важен)
    4) Вырезает ROI под дисплей и прогоняет EasyOCR
    5) Выводит распознанный текст или сообщение об ошибке
    """
    # 1) Загружаем модель:
    model = YOLO(model_path)

    # 2) Делаем предсказание (возвращается список объектов Results)
    results = model.predict(source=image_path, conf=0.10, imgsz=640, save=False)

    # 3) Создаём EasyOCR Reader (английский алфавит, цифры входят в 'en')
    reader = easyocr.Reader(['en'], gpu=False)

    # 4) Проходим по всем обнаруженным боксам
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]  # имя класса, например "Display"
            # Если нашли дисплей (не зависимо от регистра)
            if label.lower() == "Display":
                # Координаты бокса (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Загружаем изображение через OpenCV, чтобы взять ROI
                img = cv2.imread(image_path)
                if img is None:
                    print(f"ERROR: не удалось прочитать изображение по пути '{image_path}'")
                    return

                # Корректируем границы, чтобы не выйти за пределы
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    print("ERROR: некорректные координаты бокса дисплея.")
                    return

                # Вырезаем ROI (дисплей)
                roi = img[y1:y2, x1:x2]

                # Сохраняем вырезку для проверки (по желанию):
                cv2.imwrite("C:/obychenie2/cropped_display.png", roi)

                # 5) Распознаём текст через EasyOCR
                ocr_results = reader.readtext(roi)
                if not ocr_results:
                    print("Дисплей найден, но не удалось распознать текст.")
                else:
                    # Берём первый результат (предполагаем, что там одна строка)
                    _, text, prob = ocr_results[0]
                    print(f"Распознанный текст: '{text}' (достоверность {prob:.2f})")

                return  # после первого дисплея выходим

    # Если дошли до конца, а дисплей не найден
    print("Дисплей (класс 'display') не найден на изображении.")


if __name__ == "__main__":
    # Проверяем, что передан хотя бы один аргумент (путь к картинке)
    if len(sys.argv) < 2:
        print("Ошибка: укажите путь к изображению.")
        print("Пример:")
        print(r"  python read_display.py C:\obychenie2\test_images\1.jpg")
        sys.exit(1)

    # Берём путь из аргумента командной строки
    image_path = sys.argv[1]

    # Убедитесь, что файл существует
    try:
        with open(image_path, "rb"):
            pass
    except Exception:
        print(f"ERROR: не найден файл по пути '{image_path}'")
        sys.exit(1)

    # Путь к вашим весам локально (замените, если модель лежит в другом месте)
    MODEL_PATH = "C:/obychenie2/runs/detect/local_yolov11/weights/best.pt"

    read_display_from_image(image_path, MODEL_PATH)
