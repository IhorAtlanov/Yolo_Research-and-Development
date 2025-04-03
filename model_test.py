"""
Скрипт для перевірки працездатності YOLO моделі на фото та відео.
Підтримує обробку окремих зображень, відеофайлів та відео з вебкамери.
Додаткові можливості:
- Обробка кожного n-го кадру для підвищення продуктивності
- Збереження окремих кадрів з виявленнями у вигляді зображень
- Відображення середнього часу обробки всіх кадрів

Використання:
    # Для обробки зображення:
    python model_test.py --source path/to/image.jpg --model path/to/best.pt

    # Для обробки відео:
    python model_test.py --source path/to/video.mp4 --model path/to/best.pt

    # Для захоплення з веб-камери:
    python model_test.py --source 0 --model path/to/best.pt

    # Додаткові параметри:
    --conf 0.25           # Поріг впевненості (за замовчуванням 0.25)
    --iou 0.45            # Поріг IoU для NMS (за замовчуванням 0.45)
    --process-large       # Включити обробку великих зображень через розділення
    --slice-size 640      # Розмір фрагмента для розділення великих зображень
    --overlap 0.2         # Коефіцієнт перекриття для розділення
    --device 0            # Пристрій для інференсу (CPU: 'cpu', GPU: 0,1,2...)
    --save-results        # Зберігати відеорезультати
    --output-dir results  # Директорія для збереження відеорезультатів
    --frame-skip 5        # Обробка кожного n-го кадру (за замовчуванням 5)
    --save-frames         # Зберігати окремі кадри з виявленнями
    --frames-dir frames   # Директорія для збереження кадрів
"""

import os
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Детектор танків на основі YOLOv8 (покращена версія)')
    
    parser.add_argument('--source', type=str, required=True,
                        help='Шлях до зображення, відео або номер веб-камери (0, 1, 2...)')
    parser.add_argument('--model', type=str, required=True,
                        help='Шлях до моделі YOLOv8 (.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Поріг впевненості (за замовчуванням 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='Поріг IoU для NMS (за замовчуванням 0.45)')
    parser.add_argument('--device', type=str, default='0',
                        help='Пристрій для інференсу (CPU: "cpu", GPU: 0,1,2...)')
    parser.add_argument('--process-large', action='store_true',
                        help='Включити обробку великих зображень через розділення')
    parser.add_argument('--slice-size', type=int, default=640,
                        help='Розмір фрагмента для розділення великих зображень')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Коефіцієнт перекриття для розділення (0-1)')
    parser.add_argument('--save-results', action='store_true',
                        help='Зберігати відеорезультати')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Директорія для збереження відеорезультатів')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Показувати результати')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Обробка кожного n-го кадру (за замовчуванням 1 - без пропуску)')
    parser.add_argument('--save-frames', action='store_true',
                        help='Зберігати окремі кадри з виявленнями')
    parser.add_argument('--frames-dir', type=str, default='frames',
                        help='Директорія для збереження кадрів')
    
    return parser.parse_args()

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
    
    # Обробка в залежності від розміру
    height, width = image.shape[:2]
    use_slicing = args.process_large and (width > 1280 or height > 1280)
    
    if use_slicing:
        print(f"Обробка великого зображення ({width}x{height}) через розділення...")
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
    """Обробка відео або потоку з вебкамери з підтримкою пропуску кадрів і збереження окремих кадрів"""
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Обробка відео: {source_name}, розмір: {width}x{height}, FPS: {fps}")
    print(f"Обробка кожного {args.frame_skip}-го кадру")
    
    # Підготовка директорії для збереження кадрів
    if args.save_frames:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        frames_path = os.path.join(
            args.frames_dir, 
            f"{Path(args.source).stem if not args.source.isdigit() else 'webcam'}_{timestamp}"
        )
        os.makedirs(frames_path, exist_ok=True)
        print(f"Кадри будуть збережені в {frames_path}")
    
    # Підготовка для запису результату
    video_writer = None
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir, 
            f"detection_{Path(args.source).stem if not args.source.isdigit() else 'webcam'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps / args.frame_skip, (width, height))
    
    # Змінні для відстеження швидкості та статистики
    frame_count = 0
    processed_frames = 0
    skipped_frames = 0
    total_objects = 0
    processing_times = []
    start_time_total = time.time()
    last_fps_update = time.time()
    real_fps = 0
    
    # Обробка кадрів
    while True:
        frame_read_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Пропуск кадрів
        if frame_count % args.frame_skip != 0:
            skipped_frames += 1
            continue
        
        processed_frames += 1
        
        # Час початку обробки (включаючи всі операції)
        start_time = time.time()
        
        # Обробка в залежності від розміру
        use_slicing = args.process_large and (width > 1280 or height > 1280)
        
        if use_slicing:
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
            ]
        
        # Вимірювання повного часу обробки (включаючи післяобробку)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Оновлення статистики
        total_objects += len(detections)
        
        # Оновлення реальної частоти кадрів (кожні 2 секунди)
        if time.time() - last_fps_update > 2.0:
            elapsed = time.time() - last_fps_update
            frames_in_period = len(processing_times[-int(elapsed*5):])
            if frames_in_period > 0:
                real_fps = frames_in_period / elapsed
            last_fps_update = time.time()
        
        # Розрахунок швидкості обробки кадрів (модельний FPS)
        model_fps = 1 / (sum(processing_times[-min(10, len(processing_times)):]) / min(10, len(processing_times)))
        
        # Додавання інформації на кадр
        info_text = f"Frame: {frame_count} | Objects: {len(detections)} | Model Speed: {model_fps:.1f} FPS | Real: {real_fps:.1f} FPS"
        cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Збереження кадру як зображення
        if args.save_frames:
            frame_path = os.path.join(frames_path, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, result_frame)
        
        # Запис результату у відео
        if video_writer:
            video_writer.write(result_frame)
        
        # Показ результату
        if args.show:
            cv2.imshow("Result", result_frame)
            
            # Вихід при натисканні 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Розрахунок статистики
    total_time = time.time() - start_time_total
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # Звільнення ресурсів
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Обчислення реальної та теоретичної швидкості
    model_speed = 1/avg_processing_time if avg_processing_time > 0 else 0  # Швидкість обробки моделі
    effective_fps = processed_frames / total_time if total_time > 0 else 0  # Ефективна швидкість з урахуванням пропуску
    original_video_fps = fps  # Оригінальна швидкість відео
    
    # Виведення статистики
    print("\n--- Підсумок обробки відео ---")
    print(f"Загальна кількість кадрів: {frame_count}")
    print(f"Оброблено кадрів: {processed_frames}")
    print(f"Пропущено кадрів: {skipped_frames}")
    print(f"Знайдено об'єктів: {total_objects}")
    print(f"Середній час обробки кадру: {avg_processing_time:.4f} секунд")
    print(f"Загальний час обробки: {total_time:.2f} секунд")
    print(f"Оригінальна швидкість відео: {original_video_fps:.2f} FPS")
    print(f"Швидкість обробки моделі: {model_speed:.2f} FPS (теоретична)")
    if args.frame_skip > 1:
      print(f"Ефективна швидкість: {effective_fps:.2f} FPS (реальна з пропуском)")
    else:
        print(f"Ефективна швидкість: {effective_fps:.2f} FPS (у реальному світі)")
    
    if args.save_results and video_writer:
        print(f"Результат відео збережено в {output_path}")
    
    if args.save_frames:
        print(f"Кадри збережено в {frames_path}")

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