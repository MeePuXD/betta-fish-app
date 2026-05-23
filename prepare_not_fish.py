"""
ดาวน์โหลดรูป not_fish สำหรับ train โมเดลให้รู้จัก "ไม่ใช่ปลา"
"""
import os
import requests
import time

BASE_DIR = r"c:\Users\Mon\Downloads\ปลากัด"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train", "not_fish")
VAL_DIR   = os.path.join(BASE_DIR, "dataset", "val",   "not_fish")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR,   exist_ok=True)

def download_picsum(folder, count, start_seed=0):
    success = 0
    for i in range(count):
        seed = start_seed + i
        url = f"https://picsum.photos/seed/{seed}/224/224"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                path = os.path.join(folder, f"notfish_{seed:04d}.jpg")
                with open(path, "wb") as f:
                    f.write(r.content)
                success += 1
                if success % 10 == 0:
                    print(f"  ดาวน์โหลดแล้ว {success}/{count}")
        except Exception as e:
            print(f"  [skip] seed={seed}: {e}")
        time.sleep(0.05)
    return success

print("=== เตรียมรูป not_fish ===\n")

existing_train = len([f for f in os.listdir(TRAIN_DIR) if f.endswith(".jpg")])
existing_val   = len([f for f in os.listdir(VAL_DIR)   if f.endswith(".jpg")])
print(f"มีอยู่แล้ว — train: {existing_train}, val: {existing_val}")

need_train = max(0, 120 - existing_train)
need_val   = max(0, 30  - existing_val)

if need_train > 0:
    print(f"\nดาวน์โหลดรูป train อีก {need_train} รูป...")
    n = download_picsum(TRAIN_DIR, need_train, start_seed=existing_train)
    print(f"  เสร็จ: {n} รูป")

if need_val > 0:
    print(f"\nดาวน์โหลดรูป val อีก {need_val} รูป...")
    n = download_picsum(VAL_DIR, need_val, start_seed=2000 + existing_val)
    print(f"  เสร็จ: {n} รูป")

train_total = len([f for f in os.listdir(TRAIN_DIR) if f.endswith(".jpg")])
val_total   = len([f for f in os.listdir(VAL_DIR)   if f.endswith(".jpg")])
print(f"\nรวม not_fish — train: {train_total}, val: {val_total}")
print("พร้อม train แล้วครับ!")
