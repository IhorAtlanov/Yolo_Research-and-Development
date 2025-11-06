import os
from ultralytics import YOLO

def train_yolo(data_yaml_path, epochs=50, batch_size=16, img_size=640, experiment_name='test_1', lr0=0.001, lrf=0.0001):
    model = YOLO('./yolo.pt')

    # Configuring training parameters
    results = model.train(
        data=data_yaml_path,         # path to data configuration
        epochs=epochs,               # number of epochs
        batch=batch_size,            # package size
        imgsz=img_size,              # image size
        name=experiment_name,        # name of the experiment
        lr0=lr0,                     # initial learning speed
        lrf=lrf,                     # final learning speed
        optimizer='SGD',             # Specify the optimizer
        momentum=0.937,              # Specify the pulse
        save=True,                   # preservation of results
        device='0',                  # GPU utilization (if available)
        workers=6,                   # number of workers for data loading
        project='YOLO_final_training_best_model',     # project name
        exist_ok=False,              # rewrite the previous experiment
        patience=10,                 # early termination if there is no improvement after 10 epochs
        pretrained=True,             # use of a pre-trained model
        augment=True                 # on-the-fly augmentation
    )
    
    # Returning to the trained model
    return results

def evaluate_yolo_model(model_path, data_yaml_path):
    model = YOLO(model_path)
    
    # Using the val method for a test set
    results = model.val(
        data=data_yaml_path,         # Path to data configuration
        split='test',                # Specify the test set
        save=True,                   # Saving results, including the error matrix
        project='####',     # Project name for saving
        name='####'       # Directory name for results (test_evaluation)
    )
    
    # Derivation of key metrics
    print(f"mAP@0.5: {results.box.map50}")
    print(f"mAP@0.5:0.95: {results.box.map}")
    
    return results

# Example of use
if __name__ == "__main__":
    data_yaml_path = "./data.yaml"
    experiment_name = '####'

    # Model training
    results = train_yolo(
        data_yaml_path=data_yaml_path,
        epochs=100,
        batch_size=16,
        img_size=640,
        experiment_name=experiment_name,
        lr0=0.001,
        lrf=0.001
    )
    
    # The path to the best model after training
    best_model_path = os.path.join('####', experiment_name, 'weights', 'best.pt')
    
    # Model evaluation on the test set
    evaluation_results = evaluate_yolo_model(best_model_path, data_yaml_path)
    
    print("Training and assessment completed!")
    
    # Inference (object detection) on a single image
    #model = YOLO(best_model_path)
    #results = model.predict("./_.jpg", save=True, conf=0.25)
    #print(f"Detected objects: {results[0].boxes}")