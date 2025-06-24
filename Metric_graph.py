import matplotlib.pyplot as plt
import pandas as pd

def plot_accuracy_from_csv(csv_path, metric='metrics/mAP50(B)'):
    # Читання CSV файлу з результатами
    df = pd.read_csv(csv_path)
    
    # Перевірка наявності метрики у файлі
    if metric not in df.columns:
        print(f"Метрика {metric} не знайдена у файлі {csv_path}")
        return
    
    # Побудова графіку
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df[metric], marker='o', label=metric)
    plt.title(f'Залежність {metric} від епох')
    plt.xlabel('Епоха')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
     results_csv_path = "D:\\diplom\\YOLO_final_training_best_model\\yolo11n_SGD(0_001)\\results.csv"
     plot_accuracy_from_csv(results_csv_path, metric= 'metrics/mAP50(B)')