import os
from ultralytics import YOLO

def train_yolo(data_yaml_path, epochs=50, batch_size=16, img_size=640):
    model = YOLO('yolov8n.pt')  # Використання малої моделі YOLOv8n
    
    # Налаштування параметрів навчання
    results = model.train(
        data=data_yaml_path,         # шлях до конфігурації даних
        epochs=epochs,               # кількість епох
        batch=batch_size,            # розмір пакету
        imgsz=img_size,              # розмір зображення
        save=True,                   # збереження результатів
        device='0',                  # використання GPU (якщо доступний)
        workers=6,                   # кількість workers для завантаження даних
        project='yolo_training',     # ім'я проекту
        name='exp',                  # ім'я експерименту
        exist_ok=True,               # перезаписати попередній експеримент
        patience=10,                 # раннє зупинення, якщо немає покращення 20 епох
        pretrained=False,             # використання попередньо навченої моделі
        augment=True                 # аугментіція на льоту (on-the-fly)
    )
    
    # Повернення шляху до навченої моделі
    return results

def evaluate_yolo_model(model_path, data_yaml_path):
    model = YOLO(model_path)
    results = model.val(data=data_yaml_path)
    
    print(f"mAP@0.5: {results.box.map50}")
    print(f"mAP@0.5:0.95: {results.box.map}")
    
    return results

# Приклад використання
if __name__ == "__main__":
    # Шлях до файлу data.yaml
    data_yaml_path = "D:\\diplom\\dataSet_test_640\\dataset.yaml"
    
    # Тренування моделі
    results = train_yolo(
        data_yaml_path=data_yaml_path,
        epochs=50,
        batch_size=16,
        img_size=640
    )
    
    # Шлях до найкращої моделі після навчання
    best_model_path = os.path.join('yolo_training', 'exp', 'weights', 'best.pt')
    
    # Оцінка моделі на тестовому наборі
    evaluation_results = evaluate_yolo_model(best_model_path, data_yaml_path)
    
    print("Навчання та оцінка завершені!")
    
    # Інференс (виявлення об'єктів) на одному зображенні
    model = YOLO(best_model_path)
    results = model.predict("D:\\diplom\\data\\TANK\\Tanks\\19.jpg", save=True, conf=0.25)
    print(f"Виявлені об'єкти: {results[0].boxes}")