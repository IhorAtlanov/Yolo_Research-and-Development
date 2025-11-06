from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best_yolo11n.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo11n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("./best_yolo11n_saved_model/best_yolo11n_float32.tflite")

# Run inference
results = tflite_model("00001_TANK_test.jpg")