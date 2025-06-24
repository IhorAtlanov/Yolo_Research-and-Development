#!/usr/bin/env python3
"""
Оптимізований скрипт для тестування продуктивності YOLO моделі на відео
Максимально оптимізований для швидкості без візуалізації
"""

import cv2
import time
import argparse
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path


def optimize_model(model):
    """Оптимізація моделі для інференсу"""
    # Переключення в режим evaluation
    model.model.eval()
    
    # Використання GPU якщо доступно
    if torch.cuda.is_available():
        model.model.cuda()
        print(f"✓ Використовується GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠ Використовується CPU")
    
    # Оптимізація для інференсу
    torch.backends.cudnn.benchmark = True
    return model


def test_yolo_video(video_path, model_path="yolov8n.pt", conf_threshold=0.5):
    """
    Тестування YOLO моделі на відео з максимальною оптимізацією
    
    Args:
        video_path: шлях до відео файлу
        model_path: шлях до YOLO моделі
        conf_threshold: поріг впевненості для детекції
    """
    
    print(f"🚀 Завантаження моделі: {model_path}")
    
    # Завантаження та оптимізація моделі
    model = YOLO(model_path)
    model = optimize_model(model)
    
    # Відкриття відео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не вдалося відкрити відео: {video_path}")
    
    # Отримання властивостей відео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Відео: {Path(video_path).name}")
    print(f"   Кадрів: {total_frames}")
    print(f"   FPS відео: {video_fps:.1f}")
    print(f"   Поріг впевненості: {conf_threshold}")
    print("\n🔥 Початок тестування...\n")
    
    # Змінні для метрик
    frame_times = []
    processed_frames = 0
    total_detections = 0
    
    # Початковий час
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Час початку обробки кадру
            frame_start = time.time()
            
            # Інференс без збереження результатів візуалізації
            with torch.no_grad():
                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    verbose=False,  # Відключення логування
                    save=False,     # Не зберігати результати
                    show=False,     # Не показувати
                    stream=False    # Не використовувати стрім режим
                )
            
            # Підрахунок детекцій (опціонально)
            if results and len(results) > 0:
                total_detections += len(results[0].boxes) if results[0].boxes is not None else 0
            
            # Час завершення обробки кадру
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            
            processed_frames += 1
            
            # Прогресс кожні 100 кадрів
            if processed_frames % 100 == 0:
                current_fps = 1.0 / np.mean(frame_times[-100:])
                print(f"Кадр {processed_frames}/{total_frames} | FPS: {current_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n⚠ Тестування перервано користувачем")
    
    finally:
        cap.release()
    
    # Підрахунок фінальних метрик
    total_time = time.time() - start_time
    
    if frame_times:
        avg_frame_time = np.mean(frame_times)
        avg_fps = 1.0 / avg_frame_time
        min_frame_time = np.min(frame_times)
        max_frame_time = np.max(frame_times)
        std_frame_time = np.std(frame_times)
    else:
        avg_frame_time = avg_fps = min_frame_time = max_frame_time = std_frame_time = 0
    
    # Виведення результатів
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТИ ТЕСТУВАННЯ")
    print("="*60)
    print(f"Оброблено кадрів:           {processed_frames}")
    print(f"Загальний час:              {total_time:.2f} сек")
    print(f"Середній FPS:               {avg_fps:.2f}")
    print(f"Середній час кадру:         {avg_frame_time*1000:.2f} мс")
    print(f"Мін час кадру:              {min_frame_time*1000:.2f} мс")
    print(f"Макс час кадру:             {max_frame_time*1000:.2f} мс")
    print(f"Стандартне відхилення:      {std_frame_time*1000:.2f} мс")
    print(f"Загальна кількість детекцій: {total_detections}")
    print(f"Середньо детекцій на кадр:  {total_detections/processed_frames:.1f}")
    
    # Порівняння з реальним FPS відео
    if video_fps > 0:
        realtime_ratio = avg_fps / video_fps
        print(f"Коефіцієнт реального часу:  {realtime_ratio:.2f}x")
        if realtime_ratio >= 1.0:
            print("✅ Обробка в реальному часі досягнута!")
        else:
            print("⚠ Обробка повільніше реального часу")


def main():
    parser = argparse.ArgumentParser(description="YOLO відео тест продуктивності")
    parser.add_argument("video", help="Шлях до відео файлу")
    parser.add_argument("-m", "--model", default="yolov8n.pt", 
                       help="Шлях до YOLO моделі (за замовчуванням: yolov8n.pt)")
    parser.add_argument("-c", "--conf", type=float, default=0.5,
                       help="Поріг впевненості (за замовчуванням: 0.5)")
    
    args = parser.parse_args()
    
    # Перевірка існування файлів
    if not Path(args.video).exists():
        print(f"❌ Відео файл не знайдено: {args.video}")
        return
    
    try:
        test_yolo_video(args.video, args.model, args.conf)
    except Exception as e:
        print(f"❌ Помилка: {e}")


if __name__ == "__main__":
    # Приклад використання без аргументів командного рядка
    # Замініть на ваші шляхи
    VIDEO_PATH = "D:\\diplom\\video_and_photo\\BMP.MP4"  # Замініть на свій відео файл
    MODEL_PATH = "D:\\diplom\\YOLO_final_training_best_model\\yolo11n_SGD(0_001)\\weights\\yolo11n_SGD(0_001).pt"      # Можна змінити на yolov8s.pt, yolov8m.pt, тощо
    
    if Path(VIDEO_PATH).exists():
        test_yolo_video(VIDEO_PATH, MODEL_PATH, conf_threshold=0.5)
    else:
        print("Для запуску скрипта:")
        print("python script.py your_video.mp4 -m yolov8n.pt -c 0.5")
        print("\nАбо замініть VIDEO_PATH у коді на ваш відео файл")