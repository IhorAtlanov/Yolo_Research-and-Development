import cv2
import os

def create_output_folder(folder_path):
    """Создает папку для сохранения кадров, если она не существует"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Создана папка: {folder_path}")
    else:
        print(f"Папка уже существует: {folder_path}")

def extract_frames(video_path, output_folder, frame_interval=1):
    """
    Извлекает кадры из видео с заданным интервалом
    
    Args:
        video_path (str): Путь к видеофайлу
        output_folder (str): Папка для сохранения кадров
        frame_interval (int): Интервал между кадрами (1 = каждый кадр, 5 = каждый 5-й кадр)
    """
    # Проверяем существование видеофайла
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл не найден - {video_path}")
        return False
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео - {video_path}")
        return False
    
    # Получаем информацию о видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print("Информация о видео:")
    print(f"  Всего кадров: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Длительность: {duration:.2f} секунд")
    print(f"  Интервал извлечения: каждый {frame_interval}-й кадр")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Сохраняем кадр только если он соответствует интервалу
        if frame_count % frame_interval == 0:
            # Формируем имя файла
            filename = f"frame_{frame_count:06d}.jpg"
            filepath = os.path.join(output_folder, filename)
            
            # Сохраняем кадр
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:  # Показываем прогресс каждые 100 кадров
                print(f"Сохранено кадров: {saved_count}")
        
        frame_count += 1
    
    cap.release()
    
    print("Извлечение завершено!")
    print(f"Обработано кадров: {frame_count}")
    print(f"Сохранено кадров: {saved_count}")
    print(f"Кадры сохранены в: {output_folder}")
    
    return True

def get_video_info(video_path):
    """Получает информацию о видео без извлечения кадров"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео - {video_path}")
        return None
    
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    info['duration'] = info['total_frames'] / info['fps']
    
    cap.release()
    return info

def main():
    """Основная функция с примерами использования"""
    
    # Настройки (измените под свои нужды)
    video_file = "D:\\diplom\\video_and_photo\\green_tanks.mp4"  # Путь к входному видео
    output_dir = "extracted_frames"  # Папка для сохранения кадров
    interval = 25  # Каждый 5-й кадр
    
    print("=== Скрипт для извлечения кадров из видео ===\n")
    
    # Проверяем информацию о видео
    print("1. Получение информации о видео...")
    video_info = get_video_info(video_file)
    
    if video_info:
        print(f"   Разрешение: {video_info['width']}x{video_info['height']}")
        print(f"   Кадров: {video_info['total_frames']}")
        print(f"   FPS: {video_info['fps']:.2f}")
        print(f"   Длительность: {video_info['duration']:.2f} сек")
        print(f"   Будет извлечено примерно: {video_info['total_frames'] // interval} кадров\n")
    else:
        print("   Не удалось получить информацию о видео. Проверьте путь к файлу.\n")
        return
    
    # Создаем папку для кадров
    print("2. Создание выходной папки...")
    create_output_folder(output_dir)
    print()
    
    # Извлекаем кадры
    print("3. Извлечение кадров...")
    success = extract_frames(video_file, output_dir, interval)
    
    if success:
        print("\n✅ Процесс завершен успешно!")
    else:
        print("\n❌ Произошла ошибка при извлечении кадров.")

# Дополнительные примеры использования
def example_different_intervals():
    """Пример извлечения с разными интервалами"""
    video_file = "D:\\diplom\\video_and_photo\\green_tanks.mp4"
    create_output_folder("frames_every_25_Tank")
    extract_frames(video_file, "frames_every_25_Tank", 25)

if __name__ == "__main__":
    main()
    example_different_intervals()