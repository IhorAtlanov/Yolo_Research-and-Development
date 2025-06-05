"""
Скрипт для перевірки працездатності YOLO моделі на фото та відео.
Підтримує обробку окремих зображень, відеофайлів та відео з вебкамери.
Додаткові можливості:
- Обробка кожного n-го кадру для підвищення продуктивності
- Збереження окремих кадрів з виявленнями у вигляді зображень
- Відображення середнього часу обробки всіх кадрів
- Моніторинг використання пам'яті (RAM та VRAM)
- Розширена статистика та метрики

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
    --memory-monitor      # Детальний моніторинг пам'яті
"""

import os
import gc
import cv2
import time
import torch
import psutil
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

class MemoryMonitor:
    """Клас для моніторингу використання пам'яті"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available() and device != 'cpu'
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self):
        """Отримання поточного використання пам'яті"""
        memory_info = {}
        
        # Системна пам'ять (RAM)
        ram_info = self.process.memory_info()
        # Поточне використання RSS
        memory_info['ram_mb'] = ram_info.rss / 1024 / 1024
        # Відсоток використання пам'яті процесом
        memory_info['ram_percent'] = self.process.memory_percent()
        # Пік використання пам'яті (тільки на Windows)
        peak_mb = None
        if hasattr(ram_info, 'peak_wset'):
            peak_mb = ram_info.peak_wset / 1024 / 1024
        elif hasattr(ram_info, 'rss'):  # fallback для інших ОС
            peak_mb = ram_info.rss / 1024 / 1024
        memory_info['ram_peak_mb'] = peak_mb
        
        # GPU пам'ять (VRAM)
        if self.gpu_available:
            try:
                memory_info['vram_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                memory_info['vram_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
                memory_info['vram_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
            except Exception:
                memory_info['vram_allocated_mb'] = 0
                memory_info['vram_cached_mb'] = 0
                memory_info['vram_max_allocated_mb'] = 0
        
        return memory_info
    
    def get_memory_delta(self):
        """Отримання зміни пам'яті відносно початкового стану"""
        current = self.get_memory_usage()
        delta = {}
        
        for key in current:
            if key in self.initial_memory:
                delta[f"delta_{key}"] = current[key] - self.initial_memory[key]
            else:
                delta[f"delta_{key}"] = current[key]
        
        return current, delta
    
    def print_memory_info(self, title="Використання пам'яті"):
        """Виведення інформації про пам'ять"""
        current, delta = self.get_memory_delta()
        
        print(f"\n--- {title} ---")
        print(f"RAM: {current['ram_mb']:.1f} MB ({current['ram_percent']:.1f}%)")
        print(f"RAM зміна: {delta['delta_ram_mb']:+.1f} MB")
        print(f"RAM максимум: {current['ram_peak_mb']:.1f} MB")
        print(f"RAM пік зміна: {delta['delta_ram_peak_mb']:+.1f} MB")
        
        if self.gpu_available:
            print(f"VRAM виділено: {current['vram_allocated_mb']:.1f} MB")
            print(f"VRAM кешовано: {current['vram_cached_mb']:.1f} MB")
            print(f"VRAM максимум: {current['vram_max_allocated_mb']:.1f} MB")
            print(f"VRAM зміна: {delta['delta_vram_allocated_mb']:+.1f} MB")

class PerformanceTracker:
    """Клас для відстеження продуктивності"""
    
    def __init__(self):
        self.processing_times = []
        self.confidence_scores = []
        self.detection_counts = []
        self.start_time = None
        self.last_processing_time = 0  # Час останньої обробки
        
    def start_timing(self):
        """Початок вимірювання часу"""
        self.start_time = time.perf_counter()
        
    def end_timing(self, detections=None):
        """Завершення вимірювання часу"""
        if self.start_time is None:
            return 0
        
        processing_time = time.perf_counter() - self.start_time
        self.processing_times.append(processing_time)
        self.last_processing_time = processing_time  # Зберігаємо останній час
        
        if detections:
            self.detection_counts.append(len(detections))
            confidences = [d['confidence'] for d in detections]
            self.confidence_scores.extend(confidences)
        else:
            self.detection_counts.append(0)
            
        self.start_time = None
        return processing_time
    
    def get_avg_processing_time(self):
        """Середній час обробки"""
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
    
    def get_instant_fps(self):
        """Миттєва (instant) швидкість FPS за час інференсу останнього кадру."""
        return 1 / self.last_processing_time if self.last_processing_time > 0 else 0

    def get_avg_fps(self):
        """Середня швидкість FPS"""
        avg_time = self.get_avg_processing_time()
        return 1 / avg_time if avg_time > 0 else 0
    
    def get_stats(self):
        """Отримання статистики"""
        stats = {
            'last_processing_time': self.last_processing_time,
            'avg_processing_time': self.get_avg_processing_time(),
            'instant_fps': self.get_instant_fps(),
            'avg_fps': self.get_avg_fps(),          # Середній FPS
            'total_detections': sum(self.detection_counts),
            'avg_detections_per_frame': sum(self.detection_counts) / len(self.detection_counts) if self.detection_counts else 0,
            'avg_confidence': sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0,
            'min_confidence': min(self.confidence_scores) if self.confidence_scores else 0,
            'max_confidence': max(self.confidence_scores) if self.confidence_scores else 0
        }
        return stats

def get_model_info(model):
    """Отримання інформації про модель"""
    try:
        # Підрахунок параметрів моделі
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # Розмір моделі в МБ (припускаючи float32)
        model_size_mb = total_params * 4 / 1024 / 1024
        
        info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'model_type': type(model.model).__name__
        }
        
        return info
    except Exception as e:
        print(f"Помилка отримання інформації про модель: {e}")
        return {}

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
    parser.add_argument('--show', action='store_true', default=False,
                        help='Показувати результати')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Обробка кожного n-го кадру (за замовчуванням 1 - без пропуску)')
    parser.add_argument('--save-frames', action='store_true',
                        help='Зберігати окремі кадри з виявленнями')
    parser.add_argument('--frames-dir', type=str, default='frames',
                        help='Директорія для збереження кадрів')
    parser.add_argument('--memory-monitor', action='store_true',
                        help='Детальний моніторинг пам\'яті')
    
    return parser.parse_args()

def process_large_image(model, image, slice_size=640, overlap=0.2, conf=0.25, iou=0.45, tracker=None):
    """
    Обробка великого зображення шляхом розділення на фрагменти
    
    Параметри:
        model: Модель YOLO
        image: Вхідне зображення (numpy array)
        slice_size: Розмір фрагментів
        overlap: Коефіцієнт перекриття (0-1)
        conf: Поріг впевненості
        iou: Поріг IoU для NMS
        tracker: PerformanceTracker для вимірювання часу
    
    Повертає:
        Зображення з виявленими об'єктами
        Список виявлень
    """
    if tracker:
        tracker.start_timing()
        
    original_height, original_width = image.shape[:2]
    
    # Визначення кроку з урахуванням перекриття
    stride = int(slice_size * (1 - overlap))
    
    # Створення вихідного зображення та списку всіх виявлень
    result_image = image.copy()
    all_detections = []
    
    # Тимчасове ім'я файлу для збереження фрагментів
    temp_dir = os.path.join(os.getcwd(), 'temp_slices')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
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
        
        # Додавання NMS для фільтрування дублікатів
        if all_detections:
            boxes = np.array([d['box'] for d in all_detections])
            confidences = np.array([d['confidence'] for d in all_detections])
            
            # Використання NMS для фільтрування перекриваючих боксів
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf, iou)
            
            if len(indices) > 0:
                filtered_detections = [all_detections[i] for i in indices.flatten()]
            else:
                filtered_detections = []
            
            # Візуалізація виявлень
            for detection in filtered_detections:
                x1, y1, x2, y2 = [int(c) for c in detection['box']]
                confidence = detection['confidence']
                
                # Малювання боксу
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Додавання тексту
                label = f"Tank {confidence:.2f}"
                cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            final_detections = filtered_detections
        else:
            final_detections = []
            
    finally:
        # Видалення тимчасової директорії
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
    
    if tracker:
        tracker.end_timing(final_detections)
    
    return result_image, final_detections

def process_image(args, memory_monitor=None):
    """Обробка окремого зображення"""
    # Завантаження моделі
    print("Завантаження моделі...")
    model = YOLO(args.model)
    
    # Інформація про модель
    model_info = get_model_info(model)
    if model_info:
        print("\n--- Інформація про модель ---")
        print(f"Тип моделі: {model_info.get('model_type', 'Невідомо')}")
        print(f"Загальна кількість параметрів: {model_info.get('total_params', 0):,}")
        print(f"Навчальні параметри: {model_info.get('trainable_params', 0):,}")
        print(f"Розмір моделі: {model_info.get('model_size_mb', 0):.1f} MB")
    
    if memory_monitor:
        memory_monitor.print_memory_info("Після завантаження моделі")
    
    # Перевірка наявності файлу
    if not os.path.exists(args.source):
        print(f"Помилка: Файл {args.source} не знайдено")
        return
    
    # Завантаження зображення
    image = cv2.imread(args.source)
    
    # Створення трекера продуктивності
    tracker = PerformanceTracker()
    
    # Обробка в залежності від розміру
    height, width = image.shape[:2]
    use_slicing = args.process_large and (width > 1280 or height > 1280)
    
    if use_slicing:
        print(f"Обробка великого зображення ({width}x{height}) через розділення...")
        result_image, detections = process_large_image(
            model, image, args.slice_size, args.overlap, args.conf, args.iou, tracker
        )
    else:
        # Стандартний інференс
        tracker.start_timing()
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
        
        tracker.end_timing(detections)
    
    # Отримання статистики
    stats = tracker.get_stats()
    
    # Виведення інформації
    print("\n--- Результати обробки зображення ---")
    print(f"Знайдено об'єктів: {len(detections)}")
    print(f"Час обробки: {stats['avg_processing_time']:.4f} секунд")
    if detections:
        print(f"Середня впевненість: {stats['avg_confidence']:.3f}")
        print(f"Мін/макс впевненість: {stats['min_confidence']:.3f}/{stats['max_confidence']:.3f}")
    
    if memory_monitor:
        memory_monitor.print_memory_info("Після обробки зображення")
    
    # Додавання інформації на зображення
    info_text = f"Objects: {len(detections)}  Time: {stats['avg_processing_time']:.3f}s"
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

def process_video(args, memory_monitor=None):
    """Обробка відео або потоку з вебкамери з підтримкою пропуску кадрів і збереження окремих кадрів"""
    # Завантаження моделі
    print("Завантаження моделі...")
    model = YOLO(args.model)
    
    # Інформація про модель
    model_info = get_model_info(model)
    if model_info:
        print("\n--- Інформація про модель ---")
        print(f"Тип моделі: {model_info.get('model_type', 'Невідомо')}")
        print(f"Загальна кількість параметрів: {model_info.get('total_params', 0):,}")
        print(f"Розмір моделі: {model_info.get('model_size_mb', 0):.1f} MB")
    
    if memory_monitor:
        memory_monitor.print_memory_info("Після завантаження моделі")
    
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
    
    print(f"Обробка відео: {source_name}, розмір: {width}x{height}, FPS: {fps:.1f}")
    if total_frames > 0:
        print(f"Загальна кількість кадрів: {total_frames}")
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
        effective_fps = max(1, fps / args.frame_skip)
        video_writer = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))
    
    # Створення трекера продуктивності
    tracker = PerformanceTracker()
    
    # Змінні для відстеження швидкості та статистики
    frame_count = 0
    processed_frames = 0
    skipped_frames = 0
    start_time_total = time.perf_counter()
    last_stats_time = time.perf_counter()

    #DEBUG:
    timesINF = []

    # Синхронізація GPU перед початком вимірювання
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Обробка кадрів
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Пропуск кадрів
            for _ in range(args.frame_skip - 1):
                if not cap.grab():
                    break
                skipped_frames += 1
            
            processed_frames += 1
            
            # Стандартний інференс
            tracker.start_timing()
            results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
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
            
            tracker.end_timing(detections)
            
            # Отримання поточної статистики
            current_stats = tracker.get_stats()
                
            # Виведення статистики кожні 5 секунд
            current_time = time.perf_counter()
            if current_time - last_stats_time > 5.0:
                if memory_monitor:
                    memory_monitor.print_memory_info(f"Кадр {frame_count}")
                print(f"Кадр {frame_count}: FPS={current_stats['instant_fps']:.1f}, "
                      f"Об'єктів={len(detections)}, "
                      f"Час обробки={current_stats['avg_processing_time']:.4f}с")
                last_stats_time = current_time
            
            real_index = (processed_frames - 1)*args.frame_skip + 1

            # Додавання інформації на кадр
            info_text = f"Frame: {real_index} | Objects: {len(detections)} | FPS: {current_stats['instant_fps']:.1f}"
            cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Збереження кадру як зображення
            if args.save_frames:
                if len(detections) > 0:
                    frame_path = os.path.join(
                        frames_path,
                        f"frame_{real_index:06d}_det{len(detections)}.jpg"
                    )
                else:
                    frame_path = os.path.join(
                        frames_path,
                        f"frame_{real_index:06d}_no_det.jpg"
                    )
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
                
            #DEBUG:
            start = time.perf_counter()
            model.predict(frame)
            time_taken = (time.perf_counter() - start)*1000
            #print("Model-only inference time:", time_taken, "ms")
            timesINF.append(time_taken)

    except KeyboardInterrupt:
        print("\nОбробку перервано користувачем")
    
    finally:
        # Розрахунок фінальної статистики
        total_time = time.perf_counter() - start_time_total
        final_stats = tracker.get_stats()
        
        # Звільнення ресурсів
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Обчислення реальної швидкості
        effective_fps = processed_frames / total_time if total_time > 0 else 0
        
        # Виведення фінальної статистики
        print("\n" + "="*50)
        print("ПІДСУМОК ОБРОБКИ ВІДЕО")
        print("="*50)
        print(f"Загальна кількість кадрів: {frame_count:,}")
        print(f"Оброблено кадрів: {processed_frames:,}")
        print(f"Пропущено кадрів: {skipped_frames:,}")
        print(f"Знайдено об'єктів: {final_stats['total_detections']:,}")
        print(f"Середня кількість об'єктів на кадр: {final_stats['avg_detections_per_frame']:.2f}")
        
        print("\nDEBUG:")
        print(f"Загальний час: {total_time:.3f} сек")
        print(f"Оброблено кадрів: {processed_frames}")
        print(f"Розрахунок FPS: {processed_frames / total_time:.2f}")
        print(f"Час на кадр: {1000 * total_time / processed_frames:.2f} мс")

        if timesINF:
            average_time_inf_ms = sum(timesINF) / len(timesINF)
            print("Average inference time:", average_time_inf_ms, "ms")
        else:
            print("No frames processed.")
        
        average_time_inf_s = average_time_inf_ms / 1000.0
        theoretical_model_FPS = 1 / average_time_inf_s

        print("\n--- Продуктивність ---")
        print(f"Середній час обробки кадру: {final_stats['avg_processing_time']:.4f} секунд")
        print(f"Загальний час обробки: {total_time:.2f} секунд")
        print(f"Середній FPS:  {final_stats['avg_fps']:.2f} FPS")
        print(f"Миттєва швидкість (за останній оброблений кадр): {final_stats['instant_fps']:.2f} FPS")
        print(f"Ефективна швидкість: {effective_fps:.2f} FPS")
        print(f"Швидкість обробки моделі: {theoretical_model_FPS:.2f} FPS")
        print(f"Коефіцієнт прискорення: {args.frame_skip}x")
        
        if final_stats['avg_confidence'] > 0:
            print("\n--- Точність ---")
            print(f"Середня впевненість: {final_stats['avg_confidence']:.3f}")
            print(f"Мін/макс впевненість: {final_stats['min_confidence']:.3f}/{final_stats['max_confidence']:.3f}")
        
        if memory_monitor:
            memory_monitor.print_memory_info("Фінальне використання пам'яті")
        
        if args.save_results and video_writer:
            print(f"\nРезультат відео збережено в {output_path}")
        
        if args.save_frames:
            print(f"Кадри збережено в {frames_path}")
        
        # Побудова та збереження графіків продуктивності
        try:
            os.makedirs(args.output_dir, exist_ok=True)

            # 1. FPS по кадрах
            fps_values = [1/t for t in tracker.processing_times if t > 0.0]
            plt.figure(figsize=(10, 5))
            plt.plot(fps_values, label="FPS per frame")
            plt.xlabel("Оброблені кадри")
            plt.ylabel("FPS")
            plt.title("Швидкість обробки кадрів (FPS)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "fps_plot.png"))
            plt.close()

            # 2. Гістограма confidence
            plt.figure(figsize=(8, 5))
            plt.hist(tracker.confidence_scores, edgecolor='black')
            plt.xlabel("Confidence")
            plt.ylabel("Кількість")
            plt.title("Гістограма впевненості виявлень")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "confidence_histogram.png"))
            plt.close()

            print("\nГрафіки збережено у директорії:", args.output_dir)

        except Exception as e:
            print(f"Помилка при побудові графіків: {e}")

def main():
    args = parse_arguments()
    
    # Встановлення пристрою для обчислень
    if args.device.lower() != 'cpu' and torch.cuda.is_available():
        device = f"cuda:{args.device}" if args.device.isdigit() else args.device
        print(f"Використовується пристрій: {device}")
    else:
        device = "cpu"
        print(f"Використовується пристрій: {device}")
    
    # Ініціалізація моніторингу пам'яті
    memory_monitor = None
    if args.memory_monitor:
        memory_monitor = MemoryMonitor(device)
        memory_monitor.print_memory_info("Початковий стан")
        
        # Очищення пам'яті перед початком
        gc.collect()
        if device != 'cpu':
            torch.cuda.empty_cache()
    
    try:
        # Визначення типу вхідних даних
        if args.source.isdigit() or args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            process_video(args, memory_monitor)
        else:
            process_image(args, memory_monitor)
    
    except Exception as e:
        print(f"Помилка під час обробки: {e}")
        
    finally:
        # Фінальне очищення пам'яті
        if memory_monitor:
            gc.collect()
            if device != 'cpu':
                torch.cuda.empty_cache()
            memory_monitor.print_memory_info("Після очищення пам'яті")

if __name__ == "__main__":
    main()