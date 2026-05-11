from pathlib import Path
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov11n-face.pt"
INPUT_IMAGE = PROJECT_ROOT.parent.parent / "test.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output"

def main() -> None:
    model = YOLO(MODEL_PATH)
    image= cv2.imread(str(INPUT_IMAGE))
    
    results = model(image, imgsz=320, device="cpu", verbose=False)
    boxes = results[0].boxes

    print(f"Detected {len(boxes)} faces anon.")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        confidence = float(box.conf[0].item())
        print(f" face {i}: ({x1}, {y1}) -> ({x2},{y2}), conf: {confidence:.2f}")
        
        face_crop = image[y1:y2, x1:x2]
        crop_path = OUTPUT_DIR / f"face_{i}.jpg"
        cv2.imwrite(str(crop_path), face_crop)

        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        label = f"face {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    annotated_path = OUTPUT_DIR / "annotated.jpg"
    cv2.imwrite(str(annotated_path), image)
    print(f"Wrote anon.{annotated_path}")
if __name__ == "__main__":
    main()