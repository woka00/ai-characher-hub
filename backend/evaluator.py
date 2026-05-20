"""
evaluator.py — Automatic model evaluation engine
Runs YOLO, image classifiers, and text sentiment models,
then converts raw metrics into scores [1..5] for the platform.

Usage:
    python evaluator.py --project_id 1 --model_type detection
"""
import time, json, argparse, os, sys
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ── Score converters ─────────────────────────────────────────────────────────
def fps_to_score(fps: float) -> float:
    if fps >= 30: return 5.0
    if fps >= 15: return 4.0
    if fps >= 5:  return 3.0
    if fps >= 2:  return 2.0
    return 1.0

def confidence_to_score(conf: float) -> float:
    return round(conf * 5, 2)

def size_to_score(size_mb: float) -> float:
    if size_mb < 10:  return 5.0
    if size_mb < 30:  return 4.0
    if size_mb < 80:  return 3.0
    if size_mb < 200: return 2.0
    return 1.0

def accuracy_to_score(acc: float) -> float:
    """acc in [0,1]"""
    return round(acc * 5, 2)

def wer_to_score(wer: float) -> float:
    """Lower WER is better"""
    if wer < 0.05: return 5.0
    if wer < 0.15: return 4.0
    if wer < 0.30: return 3.0
    if wer < 0.50: return 2.0
    return 1.0

def noise_stability_to_score(base_det: int, noise_det: int) -> float:
    if base_det == 0: return 1.0
    ratio = noise_det / base_det
    if ratio >= 0.90: return 5.0
    if ratio >= 0.75: return 4.0
    if ratio >= 0.55: return 3.0
    if ratio >= 0.35: return 2.0
    return 1.0

# ── YOLO evaluator ────────────────────────────────────────────────────────────
def evaluate_yolo():
    """
    Returns dict: model_name -> {criterion_name -> score}
    Requires: pip install ultralytics opencv-python numpy
    """
    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
        from pathlib import Path
    except ImportError:
        print("Install: pip install ultralytics opencv-python numpy")
        return {}

    models = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt",
        "YOLOv8m": "yolov8m.pt",
    }

    # Use built-in COCO8 sample images
    try:
        from ultralytics.utils import ASSETS
        test_images = list(ASSETS.glob("*.jpg"))[:5]
        if not test_images:
            raise FileNotFoundError
    except Exception:
        print("No test images found, using synthetic noise test only")
        test_images = []

    results = {}

    for model_name, model_path in models.items():
        print(f"  Evaluating {model_name}...")
        model = YOLO(model_path)

        times, confs, base_detections, noise_detections = [], [], [], []

        for img_path in test_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Clean inference
            t0 = time.time()
            r  = model(img, verbose=False)[0]
            times.append(time.time() - t0)
            if len(r.boxes):
                confs.extend(r.boxes.conf.tolist())
            base_detections.append(len(r.boxes))

            # Noise test (robustness)
            noise = np.clip(img.astype(np.int16) + np.random.normal(0, 30, img.shape), 0, 255).astype(np.uint8)
            r_n   = model(noise, verbose=False)[0]
            noise_detections.append(len(r_n.boxes))

        avg_conf = sum(confs) / len(confs) if confs else 0.6
        avg_fps  = 1 / (sum(times) / len(times)) if times else 5.0
        avg_base = sum(base_detections) / len(base_detections) if base_detections else 1
        avg_noise= sum(noise_detections) / len(noise_detections) if noise_detections else 0

        # Model size
        import os
        try:
            size_mb = os.path.getsize(model_path) / 1e6
        except Exception:
            size_mb = {"YOLOv8n": 6, "YOLOv8s": 22, "YOLOv8m": 52}.get(model_name, 30)

        results[model_name] = {
            "Точность ответа":         confidence_to_score(avg_conf),
            "Глубина и полнота":        round(min(avg_base / 3 * 5, 5), 2),
            "Логичность и структура":  4.0,   # detection = structured output
            "Гибкость интерпретации":  3.5,   # fixed
            "Устойчивость к шуму":     noise_stability_to_score(avg_base, avg_noise),
            "Обработка сложных задач": confidence_to_score(avg_conf) * 0.9,
            "Скорость ответа":         fps_to_score(avg_fps),
            "Контекстная согласованность": 3.5,  # detectors don't have context
            "Адаптивность":            3.0,
            "Компактность модели":     size_to_score(size_mb),
        }

    return results

# ── Image classifier evaluator ────────────────────────────────────────────────
def evaluate_classifiers():
    """
    Requires: pip install torch torchvision pillow
    """
    try:
        import torch
        import torchvision.models as tvm
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        print("Install: pip install torch torchvision pillow")
        return {}

    model_configs = {
        "MobileNetV3":    tvm.mobilenet_v3_small(weights="DEFAULT"),
        "EfficientNet-B0": tvm.efficientnet_b0(weights="DEFAULT"),
        "ResNet-50":      tvm.resnet50(weights="DEFAULT"),
        "ConvNeXt-Tiny":  tvm.convnext_tiny(weights="DEFAULT"),
    }

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    import urllib.request, io
    # Download a sample image
    try:
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
        urllib.request.urlretrieve(url, "/tmp/test_img.jpg")
        test_images = ["/tmp/test_img.jpg"]
    except Exception:
        print("Could not download test image, using random tensor")
        test_images = []

    results = {}
    for name, model in model_configs.items():
        print(f"  Evaluating {name}...")
        model.eval()

        times, confs = [], []

        if test_images:
            for img_path in test_images:
                try:
                    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
                    t0  = time.time()
                    with torch.no_grad():
                        out = model(img)
                    times.append(time.time() - t0)
                    probs = torch.softmax(out, dim=1)
                    top1  = probs.max().item()
                    confs.append(top1)
                except Exception:
                    pass
        else:
            # Synthetic
            dummy = torch.randn(1, 3, 224, 224)
            t0 = time.time()
            with torch.no_grad():
                out = model(dummy)
            times.append(time.time() - t0)
            confs.append(torch.softmax(out, dim=1).max().item())

        avg_conf = sum(confs) / len(confs) if confs else 0.7
        avg_fps  = 1 / (sum(times) / len(times)) if times else 3.0

        import torchvision
        param_m = sum(p.numel() for p in model.parameters()) / 1e6
        size_mb = param_m * 4 / 1e6 * 1000  # rough

        results[name] = {
            "Точность ответа":         confidence_to_score(avg_conf),
            "Глубина и полнота":        confidence_to_score(avg_conf) * 0.95,
            "Логичность и структура":  4.5,   # classification = clear output
            "Гибкость интерпретации":  3.0,
            "Устойчивость к шуму":     confidence_to_score(avg_conf) * 0.85,
            "Обработка сложных задач": 3.0,
            "Скорость ответа":         fps_to_score(avg_fps),
            "Контекстная согласованность": 3.0,
            "Адаптивность":            3.5,
            "Компактность модели":     size_to_score(param_m),
        }

    return results

# ── Text sentiment evaluator ──────────────────────────────────────────────────
def evaluate_text_models():
    """
    Requires: pip install transformers torch
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("Install: pip install transformers torch")
        return {}

    model_configs = {
        "rubert-tiny2":     ("sentiment-analysis", "cointegrated/rubert-tiny2"),
        "rubert-sentiment": ("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment"),
        "roberta-sentiment": ("sentiment-analysis", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    }

    test_texts = [
        "Этот продукт отличного качества, очень доволен покупкой!",
        "Ужасное качество, полное разочарование.",
        "Нормально, ничего особенного.",
        "чо за хрень вообще не работает",          # noise/slang
        "Качество приемлемое но могло быть лучше",
    ]

    results = {}
    for name, (task, model_id) in model_configs.items():
        print(f"  Evaluating {name}...")
        try:
            pipe = pipeline(task, model=model_id, truncation=True)
            times, confs = [], []

            for text in test_texts:
                t0  = time.time()
                out = pipe(text)
                times.append(time.time() - t0)
                confs.append(out[0]['score'])

            avg_conf = sum(confs) / len(confs)
            avg_fps  = 1 / (sum(times) / len(times)) if times else 0.5

            results[name] = {
                "Точность ответа":         confidence_to_score(avg_conf),
                "Глубина и полнота":        confidence_to_score(avg_conf) * 0.9,
                "Логичность и структура":  4.0,
                "Гибкость интерпретации":  confidence_to_score(avg_conf) * 0.85,
                "Устойчивость к шуму":     confidence_to_score(avg_conf) * 0.8,
                "Обработка сложных задач": confidence_to_score(avg_conf) * 0.9,
                "Скорость ответа":         fps_to_score(avg_fps),
                "Контекстная согласованность": confidence_to_score(avg_conf),
                "Адаптивность":            confidence_to_score(avg_conf) * 0.85,
                "Компактность модели":     4.0 if 'tiny' in name.lower() else 3.0,
            }
        except Exception as e:
            print(f"  Failed {name}: {e}")

    return results

# ── Push scores to API ────────────────────────────────────────────────────────
def push_scores(project_id: int, eval_results: dict):
    with httpx.Client(base_url=API_BASE) as client:
        proj = client.get(f"/api/projects/{project_id}").json()
        models_map   = {m['name']: m['id'] for m in proj['models']}
        criteria_map = {c['name']: c['id'] for c in proj['criteria']}

        pushed, skipped = 0, 0
        for model_name, scores in eval_results.items():
            if model_name not in models_map:
                print(f"  Model '{model_name}' not in project — add it first")
                skipped += 1
                continue
            for crit_name, score in scores.items():
                if crit_name not in criteria_map:
                    continue
                score_val = max(1.0, min(5.0, float(score)))
                client.post(f"/api/projects/{project_id}/scores", json={
                    "model_id":     models_map[model_name],
                    "criterion_id": criteria_map[crit_name],
                    "score":        score_val,
                })
                pushed += 1

        print(f"  Pushed {pushed} scores, skipped {skipped} models")

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Model Evaluator")
    parser.add_argument("--project_id", type=int, required=True)
    parser.add_argument("--model_type", choices=["detection", "classification", "text", "all"],
                        default="all")
    parser.add_argument("--dry_run", action="store_true", help="Print scores without pushing")
    args = parser.parse_args()

    all_results = {}

    if args.model_type in ("detection", "all"):
        print("\n[1/3] Evaluating detection models (YOLO)...")
        all_results.update(evaluate_yolo())

    if args.model_type in ("classification", "all"):
        print("\n[2/3] Evaluating image classifiers...")
        all_results.update(evaluate_classifiers())

    if args.model_type in ("text", "all"):
        print("\n[3/3] Evaluating text/sentiment models...")
        all_results.update(evaluate_text_models())

    if args.dry_run:
        print("\n--- DRY RUN (scores not pushed) ---")
        print(json.dumps(all_results, ensure_ascii=False, indent=2))
    else:
        print(f"\n→ Pushing scores to project {args.project_id}...")
        push_scores(args.project_id, all_results)
        print("Done. Now run: POST /api/projects/{id}/calculate")
