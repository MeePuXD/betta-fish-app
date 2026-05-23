"""Export best model to ONNX for faster CPU inference"""
import os
from ultralytics import YOLO

BASE_DIR  = r"c:\Users\Mon\Downloads\ปลากัด"
MODEL_DIR = os.path.join(BASE_DIR, "models")

def find_best_pt():
    for name in ["betta_classifier-v7n", "betta_classifier-v7", "betta_classifier-v6"]:
        pt = os.path.join(MODEL_DIR, name, "weights", "best.pt")
        if os.path.exists(pt):
            return pt, name
    return None, None

pt, name = find_best_pt()
if not pt:
    print("ไม่พบโมเดล")
    exit(1)

print(f"Export: {pt}")
model = YOLO(pt)
model.export(format="onnx", imgsz=224, simplify=True, opset=12)

onnx_path = pt.replace(".pt", ".onnx")
print(f"ONNX saved: {onnx_path}")
