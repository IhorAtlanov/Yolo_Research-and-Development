import os
from pathlib import Path
from ultralytics import YOLO

def evaluate_yolo_model(model_path, data_yaml_path, conf_threshold=0.25, iou_threshold=0.7):
    """
    Тестування YOLO моделі на тестовому наборі даних
    
    Args:
        model_path (str): Шлях до файлу моделі (.pt)
        data_yaml_path (str): Шлях до конфігураційного файлу даних (.yaml)
        conf_threshold (float): Поріг впевненості для детекції
        iou_threshold (float): Поріг IoU для NMS
    
    Returns:
        results: Об'єкт з результатами валідації
    """
    
    # Перевірка існування файлів
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не знайдена: {model_path}")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Конфігураційний файл не знайдений: {data_yaml_path}")
    
    try:
        # Завантаження моделі
        print(f"Завантаження моделі з {model_path}")
        model = YOLO(model_path)
        
        # Тестування моделі
        print("Початок тестування...")
        results = model.val(
            data=data_yaml_path,
            split='test',
            save=True,
            project='YOLO_evaluation',
            name='test_results',
            conf=conf_threshold,
            iou=iou_threshold,
            plots=True,  # Створення графіків
            verbose=True  # Детальний вивід
        )
        
        # Виведення детальних метрик
        print("\n=== РЕЗУЛЬТАТИ ТЕСТУВАННЯ ===")
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"mAP@0.75: {results.box.map75:.4f}")
        
        # Метрики за класами (якщо доступні)
        if hasattr(results.box, 'mp') and results.box.mp is not None:
            print(f"Середня точність (Precision): {results.box.mp:.4f}")
            print(f"Середня повнота (Recall): {results.box.mr:.4f}")
        
        # Метрики за окремими класами
        if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
            print("\nМетрики за класами:")
            class_names = model.names if hasattr(model, 'names') else None
            
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
                ap50 = results.box.ap50[i] if len(results.box.ap50) > i else 0
                ap = results.box.ap[i] if len(results.box.ap) > i else 0
                print(f"  {class_name}: mAP@0.5={ap50:.4f}, mAP@0.5:0.95={ap:.4f}")
        """
        # Інформація про збережені файли
        save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
        if save_dir and save_dir.exists():
            print(f"\nРезультати збережено в: {save_dir}")
            
            # Список збережених файлів
            saved_files = list(save_dir.glob("*"))
            if saved_files:
                print("Збережені файли:")
                for file in saved_files:
                    print(f"  - {file.name}")
        """
        return results
        
    except Exception as e:
        print(f"Помилка під час тестування: {str(e)}")
        raise

def compare_models(model_paths, data_yaml_path):
    """
    Порівняння кількох моделей на одному тестовому наборі
    
    Args:
        model_paths (list): Список шляхів до моделей
        data_yaml_path (str): Шлях до конфігураційного файлу
    """
    results = {}
    
    print("=== ПОРІВНЯННЯ МОДЕЛЕЙ ===")
    for i, model_path in enumerate(model_paths):
        model_name = Path(model_path).stem
        print(f"\nТестування моделі {i+1}/{len(model_paths)}: {model_name}")
        
        try:
            result = evaluate_yolo_model(model_path, data_yaml_path)
            results[model_name] = {
                'mAP@0.5': result.box.map50,
                'mAP@0.5:0.95': result.box.map,
                'mAP@0.75': result.box.map75 if hasattr(result.box, 'map75') else 0
            }
        except Exception as e:
            print(f"Помилка при тестуванні {model_name}: {e}")
            results[model_name] = None
    
    # Виведення порівняльної таблиці
    print("\n=== ПОРІВНЯЛЬНА ТАБЛИЦЯ ===")
    print(f"{'Модель':<20} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'mAP@0.75':<10}")
    print("-" * 55)
    
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name:<20} {metrics['mAP@0.5']:<10.4f} {metrics['mAP@0.5:0.95']:<12.4f} {metrics['mAP@0.75']:<10.4f}")
        else:
            print(f"{model_name:<20} {'ПОМИЛКА':<10} {'ПОМИЛКА':<12} {'ПОМИЛКА':<10}")

# Приклад використання
if __name__ == "__main__":
    # Тестування однієї моделі
    data_yaml = "D:\\diplom\\dataSet_test_640\\data.yaml"
    
    #model_path = "D:\diplom\YOLO_final_training_best_model\yolo11n_no_pretrained_augment\weights\\best.pt"
    #results = evaluate_yolo_model(model_path, data_yaml)

    # Порівняння кількох моделей
    models_to_compare = [
        "D:\\diplom\\YOLO_final_training_best_model\\yolo11n\\weights\\yolo11n_AdamW.pt",
        "D:\\diplom\\YOLO_final_training_best_model\\yolo11n_no_pretrained\\weights\\yolo11n_no_pretrained.pt",
        "D:\\diplom\\YOLO_final_training_best_model\\yolo11n_no_pretrained(auto_AdamW)\\weights\\yolo11n_no_pretrained(auto_AdamW).pt", 
        "D:\\diplom\\YOLO_final_training_best_model\\yolo11n_no_pretrained(lrf_0.0001)\\weights\\yolo11n_no_pretrained(lrf_0.0001).pt",
        "D:\\diplom\\YOLO_final_training_best_model\\yolo11n_no_pretrained_augment\\weights\\yolo11n_no_pretrained_augment.pt",
        "D:\\diplom\\YOLO_final_training_best_model\\yolo11n_SGD\\weights\\yolo11n_SGD.pt",
        "D:\\diplom\\YOLO_final_training_best_model\\yolo11n_SGD(0_001)\\weights\\yolo11n_SGD(0_001).pt"
    ]
    
    compare_models(models_to_compare, data_yaml)
