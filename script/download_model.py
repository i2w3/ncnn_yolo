from pathlib import Path
import requests

from tqdm import tqdm
from ultralytics import YOLO


BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
MODEL_YEAR = ["v8", "11", "12"]
MODEL_TYPE = ["", "-cls", "-obb", "-seg", "-pose"]
IMAGE_SIZE = [640, 224, 1024, 640, 640]
MODEL_SIZE = "n"
SAVE_PATH  = Path("./models")
SAVE_PATH.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    for model_year in MODEL_YEAR:
        for model_type in MODEL_TYPE:
            model_name = f"yolo{model_year}{MODEL_SIZE}{model_type}.pt"
            model_path = SAVE_PATH / model_name
            if not model_path.exists():
                url = f"{BASE_URL}/{model_name}"
                print(f"[INFO] Downloading {model_name} from {url}...")
                
                r = requests.get(url, stream=True)
                total = int(r.headers.get("content-length", 0))
                if total <= 1024:
                    print(f"[WARNING] The model {model_name} may not exist the pretrained weights.")
                    continue
                with open(model_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True
                ) as bar:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            else:
                print(f"[INFO] {model_name} already exists. Skipping download.")
            # 导出模型
            model = YOLO(model_path)
            model.export(format="ncnn", imgsz=IMAGE_SIZE[MODEL_TYPE.index(model_type)], half=False)