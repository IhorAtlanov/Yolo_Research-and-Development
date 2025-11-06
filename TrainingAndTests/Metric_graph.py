import matplotlib.pyplot as plt
import pandas as pd

def plot_accuracy_from_csv(csv_path, metric='metrics/mAP50(B)'):
    # Reading a CSV file with results
    df = pd.read_csv(csv_path)
    
    # Checking for metrics in a file
    if metric not in df.columns:
        print(f"Metrics {metric} not found in file {csv_path}")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df[metric], marker='o', label=metric)
    plt.title(f'Dependence {metric} from the eras')
    plt.xlabel('Era')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
     results_csv_path = "./_.csv"
     # Example plot_accuracy_from_csv(results_csv_path, metric= 'metrics/mAP50(B)')
     plot_accuracy_from_csv(results_csv_path, metric= '####')