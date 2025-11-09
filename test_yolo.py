from ultralytics import YOLO

# 1) Загружаем модель YOLOv11‑nano (она автоматически скачается, если ещё нет)
model = YOLO("yolo11n.pt")

# 2) Делаем предсказание (list[Results])
results = model("https://ultralytics.com/images/zidane.jpg")

# 3) Показываем результаты на экране (GUI)
results[0].show()    # ← важно: results — это список, берём первый элемент

# 4) Если хотите сохранить результат в файл:
results[0].save()    # созданный файл попадёт в папку runs/detect/predict

# 5) Выводим текстом, что было найдено
print(results[0])
