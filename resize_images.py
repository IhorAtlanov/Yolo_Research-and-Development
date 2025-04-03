# python resize_images.py D:\diplom\data\BMP --output_folder D:\diplom\data\bmp_test

import os
from PIL import Image
import argparse

def resize_with_padding(image_path, output_path, target_size=(640, 640), background_color=(0, 0, 0)):
    """
    Змінює розмір зображення до заданого розміру, додаючи чорні смуги для збереження пропорцій.
    
    Args:
        image_path (str): Шлях до вхідного зображення
        output_path (str): Шлях для збереження зміненого зображення
        target_size (tuple): Цільовий розмір (ширина, висота)
        background_color (tuple): Колір фону (RGB)
    """
    # Відкриваємо зображення
    img = Image.open(image_path)
    
    # Отримуємо поточні розміри
    width, height = img.size
    
    # Визначаємо співвідношення сторін
    aspect = width / height
    target_aspect = target_size[0] / target_size[1]
    
    # Розрахунок нових розмірів зі збереженням пропорцій
    if aspect > target_aspect:
        # Зображення ширше за цільове співвідношення (додаємо вертикальні смуги)
        new_width = target_size[0]
        new_height = int(new_width / aspect)
    else:
        # Зображення вище за цільове співвідношення (додаємо горизонтальні смуги)
        new_height = target_size[1]
        new_width = int(new_height * aspect)
    
    # Змінюємо розмір зображення
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Створюємо нове зображення з чорним фоном
    padded_img = Image.new("RGB", target_size, background_color)
    
    # Обчислюємо позицію для вставки зображення (центрування)
    paste_position = (
        (target_size[0] - new_width) // 2,
        (target_size[1] - new_height) // 2
    )
    
    # Вставляємо зображення на чорний фон
    padded_img.paste(resized_img, paste_position)
    
    # Зберігаємо результат
    padded_img.save(output_path)

def process_folder(input_folder, output_folder, target_size=(640, 640)):
    """
    Обробляє всі зображення в заданій папці.
    
    Args:
        input_folder (str): Вхідна папка з зображеннями
        output_folder (str): Папка для збереження оброблених зображень
        target_size (tuple): Цільовий розмір (ширина, висота)
    """
    # Створюємо вихідну папку, якщо вона не існує
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Допустимі розширення файлів зображень
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    # Кількість оброблених зображень
    count = 0
    
    # Проходимо по всіх файлах у вхідній папці
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Пропускаємо папки та файли з недопустимими розширеннями
        if os.path.isdir(input_path):
            continue
            
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in valid_extensions:
            continue
        
        # Створюємо шлях для вихідного файлу
        output_path = os.path.join(output_folder, filename)
        
        try:
            # Змінюємо розмір та додаємо рамку
            resize_with_padding(input_path, output_path, target_size)
            count += 1
            print(f"Оброблено: {filename}")
        except Exception as e:
            print(f"Помилка при обробці {filename}: {e}")
    
    print(f"Всього оброблено {count} зображень")

if __name__ == "__main__":
    # Створюємо парсер аргументів командного рядка
    parser = argparse.ArgumentParser(description="Змінює розмір зображень до 640x640 з додаванням чорних смуг")
    parser.add_argument("input_folder", help="Шлях до папки з оригінальними зображеннями")
    parser.add_argument("--output_folder", help="Шлях до папки для збереження оброблених зображень (за замовчуванням: input_folder/resized)")
    
    args = parser.parse_args()
    
    # Якщо вихідна папка не вказана, створюємо підпапку 'resized' у вхідній папці
    if not args.output_folder:
        args.output_folder = os.path.join(args.input_folder, "resized")
    
    # Запускаємо обробку папки
    process_folder(args.input_folder, args.output_folder)