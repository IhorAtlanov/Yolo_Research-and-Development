from ultralytics import YOLO
import os
import cv2
from PIL import Image
import numpy as np

# 1. Функція для правильного обробки великих зображень
def process_large_image(image_path, model_path, slice_size=640, overlap=0.2, conf=0.25):
    """
    Розділяє велике зображення на фрагменти, виконує детекцію на кожному фрагменті,
    а потім об'єднує результати.
    
    Parameters:
    -----------
    image_path : str
        Шлях до великого зображення
    model_path : str
        Шлях до моделі YOLOv8
    slice_size : int
        Розмір фрагментів (рекомендується 640 або 1280)
    overlap : float
        Коефіцієнт перекриття між фрагментами (від 0 до 1)
    conf : float
        Поріг впевненості для детекції
    """
    # Завантаження моделі
    model = YOLO(model_path)
    
    # Завантаження зображення
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    
    print(f"Розмір оригінального зображення: {original_width}x{original_height}")
    
    # Розрахунок параметрів розділення
    stride = int(slice_size * (1 - overlap))
    
    # Створення вихідного зображення для візуалізації результатів
    result_image = image.copy()
    
    # Список для збереження всіх виявлених об'єктів
    all_detections = []
    
    # Розділення зображення та детекція на фрагментах
    for y in range(0, original_height, stride):
        for x in range(0, original_width, stride):
            # Визначення координат фрагменту
            x2 = min(x + slice_size, original_width)
            y2 = min(y + slice_size, original_height)
            
            # Врахування залишків по краях
            if x2 == original_width:
                x = max(0, x2 - slice_size)
            if y2 == original_height:
                y = max(0, y2 - slice_size)
            
            # Виділення фрагменту
            slice_img = image[y:y2, x:x2]
            
            # Тимчасове збереження фрагменту
            temp_slice_path = f"temp_slice_{x}_{y}.jpg"
            cv2.imwrite(temp_slice_path, slice_img)
            
            # Детекція на фрагменті
            results = model.predict(temp_slice_path, conf=conf, verbose=False)
            
            # Видалення тимчасового файлу
            os.remove(temp_slice_path)
            
            # Якщо є виявлення, додаємо їх із коригуванням координат
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for box, confidence, cls in zip(boxes, confidences, classes):
                    # Додавання зміщення до координат
                    adjusted_box = [
                        box[0] + x,  # x1
                        box[1] + y,  # y1
                        box[2] + x,  # x2
                        box[3] + y,  # y2
                    ]
                    
                    all_detections.append({
                        'box': adjusted_box,
                        'confidence': confidence,
                        'class': cls
                    })
    
    # Візуалізація всіх виявлень на оригінальному зображенні
    for detection in all_detections:
        x1, y1, x2, y2 = [int(c) for c in detection['box']]
        confidence = detection['confidence']
        class_id = int(detection['class'])
        
        # Малювання боксу
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Додавання тексту
        label = f"Tank {confidence:.2f}"
        cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Збереження результату
    output_path = f"large_image_result_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, result_image)
    
    print(f"Знайдено {len(all_detections)} об'єктів на зображенні")
    print(f"Результат збережено в {output_path}")
    
    return all_detections

# 2. Функція для правильного тренування з оптимізованими параметрами
def train_with_correct_hyperparams(data_yaml_path, epochs=100):
    """
    Навчання з коректними гіперпараметрами для YOLOv8
    """
    model = YOLO('yolov8n.pt')
    
    # Створення словника з параметрами
    hyp = {
        'lr0': 0.01,           # початкова швидкість навчання
        'lrf': 0.001,          # кінцева швидкість навчання
        'momentum': 0.937,     # SGD момент
        'weight_decay': 0.0005,# вага регуляризації
        'warmup_epochs': 3.0,  # епохи розігріву
        'warmup_momentum': 0.8,# початковий момент
        'warmup_bias_lr': 0.1, # початкова швидкість для bias
        'box': 7.5,            # втрата bbox
        'cls': 0.5,            # втрата класифікації 
        'dfl': 1.5,            # втрата розподілу
        'fl_gamma': 1.5,       # focal loss gamma
    }
    
    # Збереження гіперпараметрів у файл
    hyp_path = 'custom_hyp.yaml'
    with open(hyp_path, 'w') as f:
        for key, value in hyp.items():
            f.write(f"{key}: {value}\n")
    
    # Тренування з вказаними гіперпараметрами
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        patience=20,
        batch=16,
        imgsz=640,
        hyp=hyp_path,  # використання файлу з гіперпараметрами
        save=True,
        device='0',
        project='yolo_training_improved',
        name='correct_hyperparams',
        exist_ok=True,
        pretrained=True
    )
    
    return results

# 3. Функція для модифікації навчального зображення для кращого виявлення
def preprocess_training_images(dataset_path, output_path, contrast_factor=1.3, brightness_delta=20):
    """
    Модифікує всі навчальні зображення для покращення контрасту та яскравості,
    що може допомогти у виявленні танків.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Шлях до директорії з навчальними зображеннями
    train_images_dir = os.path.join(dataset_path, 'images', 'train')
    
    # Шлях до директорії з навчальними мітками
    train_labels_dir = os.path.join(dataset_path, 'labels', 'train')
    
    # Створення директорій для виходу
    output_images_dir = os.path.join(output_path, 'images', 'train')
    output_labels_dir = os.path.join(output_path, 'labels', 'train')
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Обробка всіх зображень
    for img_file in os.listdir(train_images_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(train_images_dir, img_file)
            
            # Завантаження зображення
            img = cv2.imread(img_path)
            
            # Застосування контрасту та яскравості
            img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=brightness_delta)
            
            # Збереження модифікованого зображення
            output_img_path = os.path.join(output_images_dir, img_file)
            cv2.imwrite(output_img_path, img)
            
            # Копіювання відповідного файлу з мітками
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(train_labels_dir, label_file)
            
            if os.path.exists(label_path):
                output_label_path = os.path.join(output_labels_dir, label_file)
                with open(label_path, 'r') as src, open(output_label_path, 'w') as dst:
                    dst.write(src.read())
    
    # Також необхідно скопіювати дані валідації
    val_images_dir = os.path.join(dataset_path, 'images', 'val')
    val_labels_dir = os.path.join(dataset_path, 'labels', 'val')
    
    output_val_images_dir = os.path.join(output_path, 'images', 'val')
    output_val_labels_dir = os.path.join(output_path, 'labels', 'val')
    
    os.makedirs(output_val_images_dir, exist_ok=True)
    os.makedirs(output_val_labels_dir, exist_ok=True)
    
    for img_file in os.listdir(val_images_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(val_images_dir, img_file)
            output_img_path = os.path.join(output_val_images_dir, img_file)
            
            img = cv2.imread(img_path)
            cv2.imwrite(output_img_path, img)
            
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(val_labels_dir, label_file)
            
            if os.path.exists(label_path):
                output_label_path = os.path.join(output_val_labels_dir, label_file)
                with open(label_path, 'r') as src, open(output_label_path, 'w') as dst:
                    dst.write(src.read())
    
    # Копіювання файлу data.yaml з відповідними змінами шляхів
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    output_data_yaml_path = os.path.join(output_path, 'data.yaml')
    
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data_yaml_content = f.read()
        
        # Заміна шляхів у файлі data.yaml
        data_yaml_content = data_yaml_content.replace(dataset_path, output_path)
        
        with open(output_data_yaml_path, 'w') as f:
            f.write(data_yaml_content)
    
    return output_path

# 4. Функція для інференсу з низьким порогом та NMS
def inference_with_low_threshold(model_path, image_path, conf=0.02, iou=0.3):
    """
    Виконує інференс з дуже низьким порогом впевненості та NMS для покращення виявлення
    
    Parameters:
    -----------
    model_path : str
        Шлях до моделі YOLOv8
    image_path : str
        Шлях до зображення
    conf : float
        Поріг впевненості (дуже низький для уникнення пропусків)
    iou : float
        Поріг IoU для NMS
    """
    model = YOLO(model_path)
    
    # Виконання інференсу з низьким порогом
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        save=True,
        save_conf=True,
        save_txt=True
    )
    
    return results

# Функція для кращого аналізу проблеми низького виявлення
def diagnose_detection_issues(model_path, image_path):
    """
    Аналізує можливі причини низької ефективності виявлення
    """
    # 1. Перевірка зображення
    image = Image.open(image_path)
    width, height = image.size
    
    print(f"Аналіз зображення: {image_path}")
    print(f"Розмір: {width}x{height}")
    
    # Дуже велике зображення для YOLO
    if width > 1280 or height > 1280:
        print("ПРОБЛЕМА: Зображення занадто велике для стандартного YOLO інференсу")
        print("РІШЕННЯ: Використовуйте функцію process_large_image")
    
    # 2. Основні проблеми з виявленням
    print("\nПотенційні проблеми з виявленням:")
    
    print("1. Масштаб об'єктів:")
    print("   - Танки можуть бути занадто малими відносно розміру зображення")
    print("   - РІШЕННЯ: Збільшіть розмір вхідного зображення (imgsz=1280)")
    
    print("\n2. Контраст і видимість:")
    print("   - Танки можуть мати низький контраст з фоном")
    print("   - РІШЕННЯ: Використовуйте попередню обробку зображень для покращення контрасту")
    
    print("\n3. Якість моделі:")
    print("   - Можливо, модель недостатньо натренована або потрібно більше даних")
    print("   - РІШЕННЯ: Збільшіть набір даних, використовуйте аугментацію, спробуйте більшу модель")
    
    # 3. Інференс з різними порогами
    print("\nПеревірка з дуже низьким порогом впевненості:")
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.01, verbose=False)
    
    num_detections = len(results[0].boxes)
    print(f"Виявлено об'єктів (з conf=0.01): {num_detections}")
    
    if num_detections > 0:
        print("РЕЗУЛЬТАТ: Модель виявляє об'єкти, але з низькою впевненістю")
        print("РІШЕННЯ: Необхідно покращити якість навчання для підвищення впевненості")
    else:
        print("РЕЗУЛЬТАТ: Модель зовсім не виявляє об'єкти")
        print("РІШЕННЯ: Необхідно провести ретельний аналіз навчальних даних")
    
    return results

# Приклад використання функцій
if __name__ == "__main__":
    data_yaml_path = "D:\\diplom\\dataSet_test_640\\data.yaml"
    model_path = "yolo_training/exp/weights/best.pt"
    test_image_path = "D:\\diplom\\data\\Tanks\\19.jpg"
    
    # 1. Спочатку діагностика проблем
    diagnose_detection_issues(model_path, test_image_path)
    
    # 2. Обробка великого зображення
    detections = process_large_image(test_image_path, model_path, slice_size=640, overlap=0.3, conf=0.02)
    
    # 3. Інференс з низьким порогом
    results = inference_with_low_threshold(model_path, test_image_path, conf=0.02, iou=0.3)