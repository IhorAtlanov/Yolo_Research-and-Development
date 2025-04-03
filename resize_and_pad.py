import cv2
import numpy as np

def resize_and_pad(image, target_size=(640, 640)):
    # Исходные размеры
    h, w = image.shape[:2]
    # Вычисляем коэффициент масштабирования
    scale = min(target_size[0] / w, target_size[1] / h)
    # Новый размер с сохранением пропорций
    new_w, new_h = int(w * scale), int(h * scale)
    # Изменяем размер
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Создаем пустое изображение 640x640
    padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    # Вычисляем отступы
    top = (target_size[1] - new_h) // 2
    left = (target_size[0] - new_w) // 2
    # Вставляем изображение в центр
    padded[top:top+new_h, left:left+new_w] = resized
    return padded

# Пример использования
image = cv2.imread("102.jpg")
result = resize_and_pad(image)
cv2.imwrite("result.jpg", result)