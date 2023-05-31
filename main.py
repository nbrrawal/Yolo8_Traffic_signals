from ultralytics import YOLO
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.chdir("E:/Dev/SD/traffic")
print("current location ", os.listdir(os.getcwd()))
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="config.yaml", epochs=1)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
#results = model("E:/Dev/SD/traffic/test.jpg")  # predict on an image
#results.save('signal_output.jpg')
#path = model.export(format="onnx")  # export the model to ONNX format