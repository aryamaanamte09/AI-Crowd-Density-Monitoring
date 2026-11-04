import cv2
import numpy
import torch
import ultralytics
import serial

print("Python: OK")
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("PySerial: OK")
print("CUDA available:", torch.cuda.is_available())