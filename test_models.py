import os
import numpy as np
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt

def get_model_info(model):
    """Отримання інформації про модель"""
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
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
    
def create_average_plots(all_results, save_dir, class_names=None):
    """Створення середніх графіків на основі всіх результатів"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Збираємо всі метрики
    map50_values = [r.box.map50 for r in all_results]
    map_values = [r.box.map for r in all_results]
    precision_values = [r.box.mp for r in all_results]
    recall_values = [r.box.mr for r in all_results]
    
    # Середні значення
    avg_map50 = np.mean(map50_values)
    avg_map = np.mean(map_values)
    avg_precision = np.mean(precision_values)
    avg_recall = np.mean(recall_values)
    
    # Стандартні відхилення
    std_map50 = np.std(map50_values)
    std_map = np.std(map_values)
    std_precision = np.std(precision_values)
    std_recall = np.std(recall_values)
    
    # 1. Середні значення з довірчими інтервалами
    plt.figure(figsize=(10, 6))
    
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    means = [avg_map50, avg_map, avg_precision, avg_recall]
    stds = [std_map50, std_map, std_precision, std_recall]
    
    x_pos = np.arange(len(metrics))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = plt.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.2)
    
    plt.xlabel('Метрики')
    plt.ylabel('Значення')
    plt.title('Середні метрики з стандартними відхиленнями')
    plt.xticks(x_pos, metrics, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Додаємо значення на стовпці
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'average_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Гістограми розподілу метрик
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(map50_values, bins=max(3, len(map50_values)//2), alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(avg_map50, color='red', linestyle='--', linewidth=2, label=f'Середнє: {avg_map50:.4f}')
    plt.title('Розподіл mAP@0.5')
    plt.xlabel('mAP@0.5')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(map_values, bins=max(3, len(map_values)//2), alpha=0.7, color='green', edgecolor='black')
    plt.axvline(avg_map, color='red', linestyle='--', linewidth=2, label=f'Середнє: {avg_map:.4f}')
    plt.title('Розподіл mAP@0.5:0.95')
    plt.xlabel('mAP@0.5:0.95')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist(precision_values, bins=max(3, len(precision_values)//2), alpha=0.7, color='red', edgecolor='black')
    plt.axvline(avg_precision, color='blue', linestyle='--', linewidth=2, label=f'Середнє: {avg_precision:.4f}')
    plt.title('Розподіл Precision')
    plt.xlabel('Precision')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.hist(recall_values, bins=max(3, len(recall_values)//2), alpha=0.7, color='magenta', edgecolor='black')
    plt.axvline(avg_recall, color='blue', linestyle='--', linewidth=2, label=f'Середнє: {avg_recall:.4f}')
    plt.title('Розподіл Recall')
    plt.xlabel('Recall')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'avg_map50': avg_map50,
        'avg_map': avg_map,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'std_map50': std_map50,
        'std_map': std_map,
        'std_precision': std_precision,
        'std_recall': std_recall
    }

def evaluate_yolo_model(model_path, data_yaml_path, split='test', project='YOLO_evaluation', 
                        name='evaluation', conf=0.25, iou=0.45, device='0', num_evaluations=10):
    """
    Оцінка моделі YOLO на тестовому датасеті з виводом детальних метрик, виконаною кілька разів
    """
    print("\n--- Початок оцінки моделі ---")
    print(f"Модель: {model_path}")
    print(f"Конфігурація датасету: {data_yaml_path}")
    print(f"Розділ для оцінки: {split}")
    print(f"Кількість оцінок: {num_evaluations}")
    
    if not os.path.exists(model_path):
        print(f"Помилка: Файл моделі {model_path} не знайдено")
        return None
    
    if not os.path.exists(data_yaml_path):
        print(f"Помилка: Файл конфігурації датасету {data_yaml_path} не знайдено")
        return None
    
    try:
        print("Завантаження моделі...")
        model = YOLO(model_path)
        
        model_info = get_model_info(model)
        if model_info:
            print("\n--- Інформація про модель ---")
            print(f"Тип моделі: {model_info.get('model_type', 'Невідомо')}")
            print(f"Загальна кількість параметрів: {model_info.get('total_params', 0):,}")
            print(f"Навчальні параметри: {model_info.get('trainable_params', 0):,}")
            print(f"Розмір моделі: {model_info.get('model_size_mb', 0):.1f} MB")
        
        all_results = []
        total_confusion_matrix = None
        class_names = None
        
        for i in range(num_evaluations):
            print(f"\nВиконання оцінки {i+1} з {num_evaluations}...")
            results = model.val(
                data=data_yaml_path,
                split=split,
                conf=conf,
                iou=iou,
                device=device,
                save=True,
                save_json=True,
                project=project,
                name=f"{name}_{i+1}",
                plots=True,
                verbose=True,
                exist_ok=True
            )
            
            all_results.append(results)
            print(f"   mAP@0.5: {results.box.map50:.4f}, mAP@0.5:0.95: {results.box.map:.4f}")
            
            # Ініціалізація та накопичення матриці помилок
            if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
                if total_confusion_matrix is None:
                    class_names = results.names
                    total_confusion_matrix = np.zeros_like(results.confusion_matrix.matrix)
                total_confusion_matrix += results.confusion_matrix.matrix
            elif total_confusion_matrix is None:
                 print("\nУвага: Матриця помилок не знайдена в результатах. Розрахунок розширених метрик буде пропущено.")

        
        # Створюємо директорію для збереження результатів
        save_dir = os.path.join(project, f"{name}_averaged_results")
        os.makedirs(save_dir, exist_ok=True)
        
        # Створюємо середні графіки та отримуємо статистики
        print("\nСтворення середніх графіків...")
        stats = create_average_plots(all_results, save_dir)
        
        print("\n" + "="*60)
        print("ФІНАЛЬНІ РЕЗУЛЬТАТИ ОЦІНКИ МОДЕЛІ")
        print("="*60)
        print(f"\n📊 Середні метрики ({num_evaluations} ітерацій):")
        print(f"   • mAP@0.5:      {stats['avg_map50']:.4f} ± {stats['std_map50']:.4f}")
        print(f"   • mAP@0.5:0.95: {stats['avg_map']:.4f} ± {stats['std_map']:.4f}")
        print(f"   • Precision:    {stats['avg_precision']:.4f} ± {stats['std_precision']:.4f}")
        print(f"   • Recall:       {stats['avg_recall']:.4f} ± {stats['std_recall']:.4f}")
        
        print("\n📈 Варіабельність метрик:")
        print(f"   • mAP@0.5 CV:      {(stats['std_map50']/stats['avg_map50']*100):.2f}%")
        print(f"   • mAP@0.5:0.95 CV: {(stats['std_map']/stats['avg_map']*100):.2f}%")
        print(f"   • Precision CV:    {(stats['std_precision']/stats['avg_precision']*100):.2f}%")
        print(f"   • Recall CV:       {(stats['std_recall']/stats['avg_recall']*100):.2f}%")
        
        # --- НОВИЙ БЛОК: РОЗРАХУНОК TP, FP, FN, TN ТА ACCURACY ---
        if total_confusion_matrix is not None and class_names is not None:
            print("\n" + "="*60)
            print(f"РОЗШИРЕНІ МЕТРИКИ ПО КЛАСАХ (на основі {num_evaluations} запусків)")
            print("="*60)

            all_class_metrics = {}
            num_classes = len(class_names)

            for i in range(num_classes):
                class_name = class_names[i]
                
                tp = total_confusion_matrix[i, i]
                fp = total_confusion_matrix[:, i].sum() - tp
                fn = total_confusion_matrix[i, :].sum() - tp
                tn = total_confusion_matrix.sum() - (tp + fp + fn)

                denominator = tp + tn + fp + fn
                accuracy = (tp + tn) / denominator if denominator > 0 else 0
                
                all_class_metrics[class_name] = {
                    'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
                    'Accuracy': f"{accuracy:.4f}"
                }

                print(f"\n✅ Клас: '{class_name}'")
                print(f"   - True Positives (TP):    {int(tp)}")
                print(f"   - False Positives (FP):   {int(fp)}")
                print(f"   - False Negatives (FN):   {int(fn)}")
                print(f"   - True Negatives (TN):    {int(tn)}")
                print(f"   - Accuracy:               {accuracy:.4f}")

            # Загальна точність (Overall Accuracy)
            total_tp = np.diag(total_confusion_matrix).sum()
            total_elements = total_confusion_matrix.sum()
            overall_accuracy = total_tp / total_elements if total_elements > 0 else 0

            print("\n" + "-"*60)
            print("🎯 ЗАГАЛЬНА ТОЧНІСТЬ (OVERALL ACCURACY)")
            print("   Розрахована як: sum(TP) / sum(всіх елементів матриці)")
            print(f"   Accuracy = {total_tp:.0f} / {total_elements:.0f} = {overall_accuracy:.4f}")
            print("-"*60)
            print("\n*Примітка: В задачах детекції об'єктів, 'TN' та 'Accuracy' інтерпретуються")
            print("інакше, ніж у класифікації. Тут TN – це об'єкти інших класів, що не були")
            print("помилково віднесені до поточного класу.")

            stats['per_class_metrics'] = all_class_metrics
            stats['overall_accuracy'] = overall_accuracy
        
        print(f"\n💾 Усі результати збережено в: {save_dir}")
        
        return stats
        
    except Exception as e:
        print(f"Помилка під час оцінки моделі: {str(e)}")
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Оцінка моделі YOLOv8')
    parser.add_argument('--model', type=str, required=True, help='Шлях до моделі YOLOv8 (.pt)')
    parser.add_argument('--data', type=str, required=True, help='Шлях до файлу data.yaml')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Розділ датасету для оцінки')
    parser.add_argument('--conf', type=float, default=0.25, help='Поріг впевненості')
    parser.add_argument('--iou', type=float, default=0.45, help='Поріг IoU для NMS')
    parser.add_argument('--device', type=str, default='0', help='Пристрій для інференсу (CPU: "cpu", GPU: 0,1,2...)')
    parser.add_argument('--num_evaluations', type=int, default=1, help='Кількість оцінок')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_yolo_model(
        model_path=args.model,
        data_yaml_path=args.data,
        split=args.split,
        project="YOLO_test_IMG",
        name="test",
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        num_evaluations=args.num_evaluations
    )