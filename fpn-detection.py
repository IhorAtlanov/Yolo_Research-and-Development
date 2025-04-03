# python fpn-detection.py --source D:\diplom\video_and_photo\1.mp4 --model D:\diplom\yolo_training\exp\weights\best.pt --use-fpn --fpn-scales 0.5,1.0,1.5 --save-results

import os
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import torch.nn.functional as F

def parse_arguments():
    parser = argparse.ArgumentParser(description='Детектор танків на основі YOLOv8/v11 з FPN')
    
    parser.add_argument('--source', type=str, required=True,
                        help='Шлях до зображення, відео або номер веб-камери (0, 1, 2...)')
    parser.add_argument('--model', type=str, required=True,
                        help='Шлях до моделі YOLOv8/v11 (.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Поріг впевненості (за замовчуванням 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='Поріг IoU для NMS (за замовчуванням 0.45)')
    parser.add_argument('--device', type=str, default='0',
                        help='Пристрій для інференсу (CPU: "cpu", GPU: 0,1,2...)')
    parser.add_argument('--process-large', action='store_true',
                        help='Включити обробку великих зображень через розділення')
    parser.add_argument('--use-fpn', action='store_true',
                        help='Включити Feature Pyramid Network для покращення детекції дрібних об\'єктів')
    parser.add_argument('--fpn-scales', type=str, default='0.5,1.0,1.5',
                        help='Масштаби для FPN (через кому, наприклад 0.5,1.0,1.5)')
    parser.add_argument('--slice-size', type=int, default=640,
                        help='Розмір фрагмента для розділення великих зображень')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Коефіцієнт перекриття для розділення (0-1)')
    parser.add_argument('--save-results', action='store_true',
                        help='Зберігати результати')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Директорія для збереження результатів')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Показувати результати')
    
    return parser.parse_args()

def process_fpn(model, image, scales=[0.5, 1.0, 1.5], conf=0.25, iou=0.45):
    """
    Обробка зображення з використанням Feature Pyramid Network
    
    Параметри:
        model: Модель YOLO
        image: Вхідне зображення (numpy array)
        scales: Масштаби для FPN
        conf: Поріг впевненості
        iou: Поріг IoU для NMS
    
    Повертає:
        Зображення з виявленими об'єктами
        Список виявлень
    """
    original_height, original_width = image.shape[:2]
    result_image = image.copy()
    all_detections = []
    
    # Тимчасова директорія для збереження масштабованих зображень
    temp_dir = os.path.join(os.getcwd(), 'temp_fpn')
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Застосування FPN з масштабами: {scales}")
    
    # Обробка кожного масштабу
    for i, scale in enumerate(scales):
        # Змінюємо розмір зображення відповідно до масштабу
        if scale != 1.0:
            scaled_width = int(original_width * scale)
            scaled_height = int(original_height * scale)
            scaled_img = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_img = image.copy()
            scaled_width, scaled_height = original_width, original_height
        
        # Зберігаємо масштабоване зображення
        scaled_img_path = os.path.join(temp_dir, f'scale_{i}_{scale}.jpg')
        cv2.imwrite(scaled_img_path, scaled_img)
        
        # Детекція на масштабованому зображенні
        results = model.predict(scaled_img_path, conf=conf, iou=iou, verbose=False)
        
        # Якщо є виявлення, перераховуємо координати для оригінального розміру
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, confidence, cls in zip(boxes, confidences, classes):
                # Масштабуємо назад до оригінального розміру
                adjusted_box = [
                    box[0] / scale,  # x1
                    box[1] / scale,  # y1
                    box[2] / scale,  # x2
                    box[3] / scale   # y2
                ]
                
                all_detections.append({
                    'box': adjusted_box,
                    'confidence': confidence,
                    'class': cls,
                    'scale': scale  # Зберігаємо інформацію про масштаб для аналізу
                })
    
    # Видалення тимчасової директорії
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    
    # Додавання NMS для фільтрування дублікатів
    if all_detections:
        boxes = np.array([d['box'] for d in all_detections])
        confidences = np.array([d['confidence'] for d in all_detections])
        
        # Використання NMS для фільтрування перекриваючих боксів
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf, iou)
        
        filtered_detections = [all_detections[i] for i in indices.flatten()]
        
        # Візуалізація виявлень
        for detection in filtered_detections:
            x1, y1, x2, y2 = [int(c) for c in detection['box']]
            confidence = detection['confidence']
            scale = detection['scale']
            
            # Визначаємо колір в залежності від масштабу, де був знайдений об'єкт
            if scale < 1.0:  # великі об'єкти (на зменшеному зображенні)
                color = (0, 0, 255)  # червоний
            elif scale == 1.0:  # середні об'єкти
                color = (0, 255, 0)  # зелений
            else:  # малі об'єкти (на збільшеному зображенні)
                color = (255, 0, 0)  # синій
            
            # Малювання боксу
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Додавання тексту
            label = f"Tank {confidence:.2f} (x{scale})"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image, filtered_detections
    
    return result_image, []

def process_large_image(model, image, slice_size=640, overlap=0.2, conf=0.25, iou=0.45):
    """
    Обробка великого зображення шляхом розділення на фрагменти
    
    Параметри:
        model: Модель YOLO
        image: Вхідне зображення (numpy array)
        slice_size: Розмір фрагментів
        overlap: Коефіцієнт перекриття (0-1)
        conf: Поріг впевненості
        iou: Поріг IoU для NMS
    
    Повертає:
        Зображення з виявленими об'єктами
        Список виявлень
    """
    original_height, original_width = image.shape[:2]
    
    # Визначення кроку з урахуванням перекриття
    stride = int(slice_size * (1 - overlap))
    
    # Створення вихідного зображення та списку всіх виявлень
    result_image = image.copy()
    all_detections = []
    
    # Тимчасове ім'я файлу для збереження фрагментів
    temp_dir = os.path.join(os.getcwd(), 'temp_slices')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Розділення зображення на фрагменти та обробка кожного
    slice_count = 0
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
            
            # Збереження фрагменту
            temp_slice_path = os.path.join(temp_dir, f'slice_{slice_count}.jpg')
            cv2.imwrite(temp_slice_path, slice_img)
            slice_count += 1
            
            # Детекція на фрагменті
            results = model.predict(temp_slice_path, conf=conf, iou=iou, verbose=False)
            
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
    
    # Видалення тимчасової директорії
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    
    # Додавання NMS для фільтрування дублікатів
    if all_detections:
        boxes = np.array([d['box'] for d in all_detections])
        confidences = np.array([d['confidence'] for d in all_detections])
        
        # Використання NMS для фільтрування перекриваючих боксів
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf, iou)
        
        filtered_detections = [all_detections[i] for i in indices.flatten()]
        
        # Візуалізація виявлень
        for detection in filtered_detections:
            x1, y1, x2, y2 = [int(c) for c in detection['box']]
            confidence = detection['confidence']
            
            # Малювання боксу
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Додавання тексту
            label = f"Tank {confidence:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image, filtered_detections
    
    return result_image, []

def process_image(args):
    """Обробка окремого зображення"""
    # Завантаження моделі
    model = YOLO(args.model)
    
    # Перевірка наявності файлу
    if not os.path.exists(args.source):
        print(f"Помилка: Файл {args.source} не знайдено")
        return
    
    # Завантаження зображення
    image = cv2.imread(args.source)
    
    start_time = time.time()
    
    # Обробка за допомогою Feature Pyramid Network
    if args.use_fpn:
        scales = [float(s) for s in args.fpn_scales.split(',')]
        result_image, detections = process_fpn(
            model, image, scales, args.conf, args.iou
        )
    # Обробка в залежності від розміру
    elif args.process_large and (image.shape[1] > 1280 or image.shape[0] > 1280):
        print(f"Обробка великого зображення ({image.shape[1]}x{image.shape[0]}) через розділення...")
        result_image, detections = process_large_image(
            model, image, args.slice_size, args.overlap, args.conf, args.iou
        )
    else:
        # Стандартний інференс
        results = model.predict(args.source, conf=args.conf, iou=args.iou)
        
        # Отримання зображення з виявленнями
        result_image = results[0].plot()
        
        # Створення списку виявлень
        boxes = results[0].boxes
        detections = [
            {
                'box': box.xyxy.cpu().numpy()[0],
                'confidence': box.conf.cpu().numpy()[0],
                'class': box.cls.cpu().numpy()[0]
            }
            for box in boxes
        ]
    
    processing_time = time.time() - start_time
    
    # Виведення інформації
    print(f"Знайдено {len(detections)} об'єктів")
    print(f"Час обробки: {processing_time:.2f} секунд")
    
    # Додавання інформації на зображення
    info_text = f"Objects: {len(detections)}  Time: {processing_time:.2f}s"
    cv2.putText(result_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Збереження результатів
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir, 
            f"detection_{Path(args.source).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(output_path, result_image)
        print(f"Результат збережено в {output_path}")
    
    # Показ результату
    if args.show:
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_image, detections

def process_video(args):
    """Обробка відео або потоку з вебкамери"""
    # Завантаження моделі
    model = YOLO(args.model)
    
    # Визначення джерела відео
    try:
        if args.source.isdigit():
            cap = cv2.VideoCapture(int(args.source))
            source_name = f"Веб-камера {args.source}"
        else:
            if not os.path.exists(args.source):
                print(f"Помилка: Файл {args.source} не знайдено")
                return
            cap = cv2.VideoCapture(args.source)
            source_name = os.path.basename(args.source)
    except Exception as e:
        print(f"Помилка відкриття відео: {e}")
        return
    
    # Перевірка успішного відкриття відео
    if not cap.isOpened():
        print("Помилка: Не вдалося відкрити відео")
        return
    
    # Отримання інформації про відео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Обробка відео: {source_name}, розмір: {width}x{height}, FPS: {fps}")
    
    # Визначення методу обробки
    processing_mode = "FPN" if args.use_fpn else ("Slicing" if args.process_large else "Standard")
    print(f"Режим обробки: {processing_mode}")
    
    # Підготовка для запису результату
    video_writer = None
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir, 
            f"detection_{Path(args.source).stem if not args.source.isdigit() else 'webcam'}_{processing_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Змінні для відстеження швидкості
    frame_count = 0
    total_time = 0
    fps_display = 0
    
    # Підготовка масштабів для FPN, якщо використовується
    if args.use_fpn:
        scales = [float(s) for s in args.fpn_scales.split(',')]
    
    # Обробка кадрів
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Вибір методу обробки
        if args.use_fpn:
            # Обробка через Feature Pyramid Network
            result_frame, detections = process_fpn(
                model, frame, scales, args.conf, args.iou
            )
        elif args.process_large and (width > 1280 or height > 1280):
            # Обробка через розділення на фрагменти
            result_frame, detections = process_large_image(
                model, frame, args.slice_size, args.overlap, args.conf, args.iou
            )
        else:
            # Стандартний інференс
            results = model.predict(frame, conf=args.conf, iou=args.iou)
            result_frame = results[0].plot()
            
            # Створення списку виявлень
            boxes = results[0].boxes
            detections = [
                {
                    'box': box.xyxy.cpu().numpy()[0],
                    'confidence': box.conf.cpu().numpy()[0],
                    'class': box.cls.cpu().numpy()[0]
                }
                for box in boxes
            ] if len(results[0].boxes) > 0 else []
        
        # Розрахунок FPS
        processing_time = time.time() - start_time
        total_time += processing_time
        if frame_count % 10 == 0:
            fps_display = 10 / total_time
            total_time = 0
        
        # Додавання інформації на кадр
        info_text = f"Objects: {len(detections)}  FPS: {fps_display:.1f} Mode: {processing_mode}"
        cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Запис результату
        if video_writer:
            video_writer.write(result_frame)
        
        # Показ результату
        if args.show:
            cv2.imshow("Result", result_frame)
            
            # Вихід при натисканні 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Звільнення ресурсів
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Обробка відео завершена. Середній FPS: {frame_count / (total_time+0.001):.1f}")
    
    if args.save_results and video_writer:
        print(f"Результат збережено в {output_path}")

def main():
    args = parse_arguments()
    
    # Встановлення пристрою для обчислень
    if args.device.lower() != 'cpu' and torch.cuda.is_available():
        device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    else:
        device = "cpu"
    
    print(f"Використовується пристрій: {device}")
    
    # Визначення типу вхідних даних
    if args.source.isdigit() or args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(args)
    else:
        process_image(args)

if __name__ == "__main__":
    main()