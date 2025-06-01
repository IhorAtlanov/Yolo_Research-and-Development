import os
from ultralytics import YOLO

def train_yolo(data_yaml_path, epochs=50, batch_size=16, img_size=640, experiment_name='test_1', lr0=0.001, lrf=0.0001):
    model = YOLO('yolo11n.pt')
    
    # Налаштування параметрів навчання
    results = model.train(
        data=data_yaml_path,         # шлях до конфігурації даних
        epochs=epochs,               # кількість епох
        batch=batch_size,            # розмір пакету
        imgsz=img_size,              # розмір зображення
        name=experiment_name,        # ім'я експерименту
        lr0=lr0,                     # початкова швидкість навчання
        lrf=lrf,                     # кінцева швидкість навчання
        optimizer='SGD',             # Вказуємо оптимізатор
        #momentum=0.937,              # Вказуємо імпульс
        save=True,                   # збереження результатів
        device='0',                  # використання GPU (якщо доступний)
        workers=6,                   # кількість workers для завантаження даних
        project='YOLO_final_training_best_model',     # ім'я проекту
        exist_ok=False,              # перезаписати попередній експеримент
        patience=10,                 # раннє зупинення, якщо немає покращення 10 епох
        pretrained=True,             # використання попередньо навченої моделі
        augment=True                 # аугментіція на льоту (on-the-fly)
    )
    
    # Повернення шляху до навченої моделі
    return results

def evaluate_yolo_model(model_path, data_yaml_path):
    model = YOLO(model_path)
    
    # Використання методу val для тестового набору
    results = model.val(
        data=data_yaml_path,         # Шлях до конфігурації даних
        split='test',                # Вказуємо тестовий набір
        save=True,                   # Збереження результатів, включаючи матрицю помилок
        project='YOLO_final_training_best_model',     # Ім'я проекту для збереження
        name='test_yolo11n_SGD(0_001)'       # Ім'я директорії для результатів (test_evaluation)
    )
    
    # Виведення основних метрик
    print(f"mAP@0.5: {results.box.map50}")
    print(f"mAP@0.5:0.95: {results.box.map}")
    
    return results

# Приклад використання
if __name__ == "__main__":
    # Шлях до файлу data.yaml
    data_yaml_path = "D:\\diplom\\dataSet_test_640\\data.yaml"

    experiment_name = 'yolo11n_SGD(0_001)'

    # Тренування моделі
    results = train_yolo(
        data_yaml_path=data_yaml_path,
        epochs=100,
        batch_size=16,
        img_size=640,
        experiment_name=experiment_name,
        lr0=0.001,
        lrf=0.001
    )
    
    # Шлях до найкращої моделі після навчання
    best_model_path = os.path.join('YOLO_final_training_best_model', experiment_name, 'weights', 'best.pt')
    
    # Оцінка моделі на тестовому наборі
    evaluation_results = evaluate_yolo_model(best_model_path, data_yaml_path)
    
    print("Навчання та оцінка завершені!")
    
    # Інференс (виявлення об'єктів) на одному зображенні
    #model = YOLO(best_model_path)
    #results = model.predict("D:\\diplom\\data\\TANK\\Tanks\\19.jpg", save=True, conf=0.25)
    #print(f"Виявлені об'єкти: {results[0].boxes}")