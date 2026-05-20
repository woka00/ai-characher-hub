"""
seed_demo.py — Populate a demo project with models, criteria, and realistic scores.
Run BEFORE the hackathon demo so calculation is instant.

Usage:
    python seed_demo.py
"""
import httpx, json, random

API = "http://localhost:8000"

def main():
    client = httpx.Client(base_url=API, timeout=10)

    # ── Create project ────────────────────────────────────────────────────
    print("Creating demo project...")
    resp = client.post("/api/projects", json={
        "name": "Сравнение моделей детекции объектов",
        "description": "Выбор оптимальной модели для задачи детекции в системе контроля качества производства"
    })
    resp.raise_for_status()
    pid = resp.json()['id']
    print(f"  Project ID: {pid}")

    # ── Add models ────────────────────────────────────────────────────────
    models_config = [
        {"name": "YOLOv8n",      "model_type": "detection",       "description": "YOLO v8 Nano — ультракомпактная"},
        {"name": "YOLOv8s",      "model_type": "detection",       "description": "YOLO v8 Small — баланс"},
        {"name": "YOLOv8m",      "model_type": "detection",       "description": "YOLO v8 Medium — точность"},
        {"name": "EfficientDet", "model_type": "detection",       "description": "EfficientDet-D0 — Google"},
        {"name": "MobileNetV3",  "model_type": "classification",  "description": "MobileNet V3 Small"},
    ]

    model_ids = {}
    print("Adding models...")
    for m in models_config:
        r = client.post(f"/api/projects/{pid}/models", json=m)
        r.raise_for_status()
        model_ids[m['name']] = r.json()['id']
        print(f"  + {m['name']}: id={model_ids[m['name']]}")

    # ── Get criteria ──────────────────────────────────────────────────────
    criteria_list = client.get(f"/api/projects/{pid}/criteria").json()
    crit_ids = {c['name']: c['id'] for c in criteria_list}
    print(f"Criteria loaded: {list(crit_ids.keys())}")

    # ── Realistic scores ──────────────────────────────────────────────────
    # Based on actual benchmarks (approximate)
    scores_matrix = {
        "YOLOv8n": {
            "Точность ответа":              3.5,
            "Глубина и полнота":            3.2,
            "Логичность и структура":       4.5,
            "Гибкость интерпретации":       3.0,
            "Устойчивость к шуму":          3.8,
            "Обработка сложных задач":      3.0,
            "Скорость ответа":              5.0,
            "Контекстная согласованность":  3.0,
            "Адаптивность":                 3.5,
            "Компактность модели":          5.0,
        },
        "YOLOv8s": {
            "Точность ответа":              4.2,
            "Глубина и полнота":            4.0,
            "Логичность и структура":       4.5,
            "Гибкость интерпретации":       3.5,
            "Устойчивость к шуму":          4.2,
            "Обработка сложных задач":      3.8,
            "Скорость ответа":              4.2,
            "Контекстная согласованность":  3.5,
            "Адаптивность":                 4.0,
            "Компактность модели":          4.0,
        },
        "YOLOv8m": {
            "Точность ответа":              4.8,
            "Глубина и полнота":            4.6,
            "Логичность и структура":       4.5,
            "Гибкость интерпретации":       4.0,
            "Устойчивость к шуму":          4.7,
            "Обработка сложных задач":      4.5,
            "Скорость ответа":              2.8,
            "Контекстная согласованность":  4.0,
            "Адаптивность":                 4.5,
            "Компактность модели":          2.5,
        },
        "EfficientDet": {
            "Точность ответа":              4.4,
            "Глубина и полнота":            4.2,
            "Логичность и структура":       4.0,
            "Гибкость интерпретации":       3.8,
            "Устойчивость к шуму":          4.0,
            "Обработка сложных задач":      4.0,
            "Скорость ответа":              3.5,
            "Контекстная согласованность":  3.8,
            "Адаптивность":                 3.8,
            "Компактность модели":          3.5,
        },
        "MobileNetV3": {
            "Точность ответа":              3.8,
            "Глубина и полнота":            3.5,
            "Логичность и структура":       4.0,
            "Гибкость интерпретации":       3.0,
            "Устойчивость к шуму":          3.5,
            "Обработка сложных задач":      3.0,
            "Скорость ответа":              4.8,
            "Контекстная согласованность":  3.0,
            "Адаптивность":                 3.5,
            "Компактность модели":          4.8,
        },
    }

    print("Pushing scores...")
    pushed = 0
    for model_name, crit_scores in scores_matrix.items():
        mid = model_ids.get(model_name)
        if not mid:
            continue
        for crit_name, score in crit_scores.items():
            cid = crit_ids.get(crit_name)
            if not cid:
                continue
            r = client.post(f"/api/projects/{pid}/scores", json={
                "model_id":     mid,
                "criterion_id": cid,
                "score":        score,
            })
            if r.status_code == 200:
                pushed += 1
    print(f"  Pushed {pushed} scores")

    # ── Calculate ─────────────────────────────────────────────────────────
    print("Running calculation...")
    resp = client.post(f"/api/projects/{pid}/calculate")
    resp.raise_for_status()
    results = resp.json()

    print("\n═══ RESULTS ═══")
    sorted_results = sorted(results.values(), key=lambda x: x['rank'])
    for m in sorted_results:
        print(f"  #{m['rank']} {m['model_name']:20s}  K={m['k']:.4f}  {m['label']}")

    print(f"\nDemo project ready at: http://localhost:8000 (project ID={pid})")
    client.close()

if __name__ == "__main__":
    main()
