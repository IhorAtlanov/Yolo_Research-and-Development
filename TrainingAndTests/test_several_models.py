import os
from pathlib import Path
from ultralytics import YOLO

def evaluate_yolo_model(model_path, data_yaml_path, conf_threshold=0.25, iou_threshold=0.7):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {data_yaml_path}")
    
    try:
        print(f"Завантаження моделі з {model_path}")
        model = YOLO(model_path)

        print("Початок тестування...")
        results = model.val(
            data=data_yaml_path,
            split='test',
            save=True,
            project='YOLO_evaluation',
            name='test_results',
            conf=conf_threshold,
            iou=iou_threshold,
            plots=True,
            verbose=True
        )

        print("\n=== TEST RESULTS ===")
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"mAP@0.75: {results.box.map75:.4f}")

        if hasattr(results.box, 'mp') and results.box.mp is not None:
            print(f"Average accuracy (Precision): {results.box.mp:.4f}")
            print(f"Average fullness (Recall): {results.box.mr:.4f}")

        if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
            print("\nMetrics by class:")
            class_names = model.names if hasattr(model, 'names') else None
            
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
                ap50 = results.box.ap50[i] if len(results.box.ap50) > i else 0
                ap = results.box.ap[i] if len(results.box.ap) > i else 0
                print(f"  {class_name}: mAP@0.5={ap50:.4f}, mAP@0.5:0.95={ap:.4f}")
        """
        # Information about saved files
        save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
        if save_dir and save_dir.exists():
            print(f"\nResults saved in: {save_dir}")
            
            # List of saved files
            saved_files = list(save_dir.glob("*"))
            if saved_files:
                print("Saved files:")
                for file in saved_files:
                    print(f"  - {file.name}")
        """
        return results
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

def compare_models(model_paths, data_yaml_path):
    results = {}
    
    print("=== COMPARISON OF MODELS ===")
    for i, model_path in enumerate(model_paths):
        model_name = Path(model_path).stem
        print(f"\nModel testing {i+1}/{len(model_paths)}: {model_name}")
        
        try:
            result = evaluate_yolo_model(model_path, data_yaml_path)
            results[model_name] = {
                'mAP@0.5': result.box.map50,
                'mAP@0.5:0.95': result.box.map,
                'mAP@0.75': result.box.map75 if hasattr(result.box, 'map75') else 0
            }
        except Exception as e:
            print(f"Error during testing {model_name}: {e}")
            results[model_name] = None

    print("\n=== COMPARATIVE TABLE ===")
    print(f"{'Model':<20} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'mAP@0.75':<10}")
    print("-" * 55)
    
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name:<20} {metrics['mAP@0.5']:<10.4f} {metrics['mAP@0.5:0.95']:<12.4f} {metrics['mAP@0.75']:<10.4f}")
        else:
            print(f"{model_name:<20} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10}")

# Example of use
if __name__ == "__main__":
    data_yaml = "./data.yaml"

    # Example
    """
        models_to_compare = [
        "./yolo11n_AdamW.pt",
        "./yolo11n_no_pretrained.pt",
        "./yolo11n_no_pretrained(auto_AdamW).pt", 
        "./yolo11n_SGD(0_001).pt"
    ]
    """

    models_to_compare = [

    ]
    
    compare_models(models_to_compare, data_yaml)
