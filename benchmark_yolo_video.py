import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np

def benchmark_yolo_video(model_path, video_path, num_frames=500):
    """
    Правильне тестування продуктивності YOLO на відео
    """
    
    # Завантаження моделі
    model = YOLO(model_path)
    
    # Відкриття відео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Помилка відкриття відео")
        return
    
    # Отримання інформації про відео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Відео: {width}x{height}, {fps} FPS, {total_frames} кадрів")
    
    # Warmup - пропуск перших кадрів для стабілізації
    warmup_frames = 10
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if ret:
            model(frame, verbose=False)
    
    # Синхронізація GPU перед початком вимірювання
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Основне тестування
    frame_times = []
    inference_times = []
    
    frames_processed = 0
    start_total = time.time()
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Кінець відео на кадрі {i}")
            break
        
        # Вимірювання часу одного кадру
        frame_start = time.time()
        
        # Вимірювання часу інференсу моделі
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Синхронізація перед початком
        
        inference_start = time.time()
        results = model(frame, verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Синхронізація після закінчення
        
        inference_end = time.time()
        frame_end = time.time()
        
        # Збереження часів
        inference_time = (inference_end - inference_start) * 1000  # в мілісекундах
        frame_time = (frame_end - frame_start) * 1000
        
        inference_times.append(inference_time)
        frame_times.append(frame_time)
        frames_processed += 1
        
        # Прогрес
        if (i + 1) % 20 == 0:
            print(f"Оброблено {i + 1}/{num_frames} кадрів...")
    
    end_total = time.time()
    total_time = end_total - start_total
    
    cap.release()
    
    # Розрахунок статистики (пропускаємо перші 5 кадрів для стабільності)
    stable_inference_times = inference_times[5:]
    stable_frame_times = frame_times[5:]
    
    avg_inference_time = np.mean(stable_inference_times)
    avg_frame_time = np.mean(stable_frame_times)
    
    # Розрахунок FPS
    theoretical_fps = 1000 / avg_inference_time  # Базується на часі інференсу
    actual_fps = frames_processed / total_time    # Реальний FPS з урахуванням всього
    
    # Виведення результатів
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТИ ТЕСТУВАННЯ ПРОДУКТИВНОСТІ")
    print("="*60)
    print(f"Кадрів оброблено: {frames_processed}")
    print(f"Загальний час: {total_time:.2f} сек")
    print()
    print("ЧАСИ (середні):")
    print(f"  Час інференсу моделі: {avg_inference_time:.2f} мс")
    print(f"  Час обробки кадру:    {avg_frame_time:.2f} мс")
    print()
    print("FPS:")
    print(f"  Теоретичний FPS (тільки модель): {theoretical_fps:.2f}")
    print(f"  Реальний FPS (весь процес):      {actual_fps:.2f}")
    print()
    print("ДЕТАЛІ:")
    print(f"  Min inference time: {min(stable_inference_times):.2f} мс")
    print(f"  Max inference time: {max(stable_inference_times):.2f} мс")
    print(f"  Std inference time: {np.std(stable_inference_times):.2f} мс")
    
    return {
        'avg_inference_time': avg_inference_time,
        'theoretical_fps': theoretical_fps,
        'actual_fps': actual_fps,
        'frames_processed': frames_processed
    }

def benchmark_yolo_images(model_path, image_size=(1920, 1080), num_iterations=100):
    """
    Тестування на синтетичних зображеннях заданого розміру
    """
    
    model = YOLO(model_path)
    
    # Створення тестового зображення
    test_image = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
    
    print(f"Тестування на зображенні {image_size[0]}x{image_size[1]}")
    
    # Warmup
    for _ in range(10):
        model(test_image, verbose=False)
    
    # Синхронізація GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Основне тестування
    times = []
    
    for i in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        results = model(test_image, verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        times.append((end - start) * 1000)
        
        if (i + 1) % 20 == 0:
            print(f"Ітерація {i + 1}/{num_iterations}")
    
    # Статистика (пропускаємо перші 5 для стабільності)
    stable_times = times[5:]
    avg_time = np.mean(stable_times)
    fps = 1000 / avg_time
    
    print(f"\nРезультати для {image_size[0]}x{image_size[1]}:")
    print(f"Середній час інференсу: {avg_time:.2f} мс")
    print(f"Теоретичний FPS: {fps:.2f}")
    print(f"Min/Max: {min(stable_times):.2f}/{max(stable_times):.2f} мс")
    
    return avg_time, fps

# Приклад використання
if __name__ == "__main__":
    MODEL_PATH = "best_yolo11n.pt"  # Змініть на ваш шлях до моделі
    VIDEO_PATH = "D:\\diplom\\video_and_photo\\Invaders.mp4"  # Змініть на ваш шлях до відео
    
    print("Тестування на відео:")
    video_results = benchmark_yolo_video(MODEL_PATH, VIDEO_PATH, num_frames=500)
    
    print("\n" + "="*60)
    print("Тестування на синтетичних зображеннях:")
    
    # Тестування різних розмірів
    sizes = [(640, 480), (1280, 720), (1920, 1080)]
    for size in sizes:
        benchmark_yolo_images(MODEL_PATH, size, num_iterations=50)
        print()