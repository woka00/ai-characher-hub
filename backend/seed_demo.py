"""
seed_demo.py — Two realistic demo projects for the defense.
Run this BEFORE your presentation.

Usage:
    python seed_demo.py
"""
import httpx, sys, os

API = os.getenv("API_BASE", "http://localhost:8000")

SCORES_DETECTION = {
  'YOLOv8n':        {'Точность ответа':3.4,'Глубина и полнота':3.1,'Логичность и структура':4.5,'Гибкость интерпретации':3.0,'Устойчивость к шуму':3.6,'Обработка сложных задач':2.9,'Скорость ответа':5.0,'Контекстная согласованность':3.0,'Адаптивность':3.5,'Компактность модели':5.0},
  'YOLOv8s':        {'Точность ответа':4.1,'Глубина и полнота':3.9,'Логичность и структура':4.5,'Гибкость интерпретации':3.5,'Устойчивость к шуму':4.1,'Обработка сложных задач':3.7,'Скорость ответа':4.2,'Контекстная согласованность':3.5,'Адаптивность':4.0,'Компактность модели':4.0},
  'YOLOv8m':        {'Точность ответа':4.8,'Глубина и полнота':4.6,'Логичность и структура':4.5,'Гибкость интерпретации':4.1,'Устойчивость к шуму':4.7,'Обработка сложных задач':4.5,'Скорость ответа':2.6,'Контекстная согласованность':4.0,'Адаптивность':4.5,'Компактность модели':2.3},
  'EfficientDet-D0':{'Точность ответа':4.3,'Глубина и полнота':4.1,'Логичность и структура':4.0,'Гибкость интерпретации':3.7,'Устойчивость к шуму':4.0,'Обработка сложных задач':3.9,'Скорость ответа':3.3,'Контекстная согласованность':3.7,'Адаптивность':3.8,'Компактность модели':3.4},
  'YOLOv5su':       {'Точность ответа':3.9,'Глубина и полнота':3.7,'Логичность и структура':4.0,'Гибкость интерпретации':3.4,'Устойчивость к шуму':3.8,'Обработка сложных задач':3.5,'Скорость ответа':4.0,'Контекстная согласованность':3.3,'Адаптивность':3.6,'Компактность модели':3.8},
}
SCORES_NLP = {
  'rubert-tiny2':      {'Точность ответа':3.6,'Глубина и полнота':3.3,'Логичность и структура':4.0,'Гибкость интерпретации':3.4,'Устойчивость к шуму':3.7,'Обработка сложных задач':3.2,'Скорость ответа':5.0,'Контекстная согласованность':4.2,'Адаптивность':4.0,'Компактность модели':5.0},
  'rubert-base':       {'Точность ответа':4.7,'Глубина и полнота':4.5,'Логичность и структура':4.5,'Гибкость интерпретации':4.3,'Устойчивость к шуму':4.5,'Обработка сложных задач':4.4,'Скорость ответа':2.4,'Контекстная согласованность':4.8,'Адаптивность':4.5,'Компактность модели':2.5},
  'roberta-sentiment': {'Точность ответа':4.4,'Глубина и полнота':4.1,'Логичность и структура':4.3,'Гибкость интерпретации':4.0,'Устойчивость к шуму':4.2,'Обработка сложных задач':4.0,'Скорость ответа':3.2,'Контекстная согласованность':4.3,'Адаптивность':4.0,'Компактность модели':3.2},
  'distilbert-ru':     {'Точность ответа':4.1,'Глубина и полнота':3.8,'Логичность и структура':4.2,'Гибкость интерпретации':3.8,'Устойчивость к шуму':3.9,'Обработка сложных задач':3.7,'Скорость ответа':4.1,'Контекстная согласованность':4.1,'Адаптивность':3.9,'Компактность модели':4.0},
}

def seed(c, name, desc, models, scores):
    r = c.post('/api/projects', json={'name': name, 'description': desc})
    r.raise_for_status()
    pid = r.json()['id']
    for m in models:
        c.post(f'/api/projects/{pid}/models', json=m).raise_for_status()
    crit = {x['name']: x['id'] for x in c.get(f'/api/projects/{pid}/criteria').json()}
    mods = {x['name']: x['id'] for x in c.get(f'/api/projects/{pid}/models').json()}
    for mn, sc in scores.items():
        mid = mods.get(mn)
        if not mid: continue
        for cn, sv in sc.items():
            cid = crit.get(cn)
            if cid: c.post(f'/api/projects/{pid}/scores', json={'model_id': mid, 'criterion_id': cid, 'score': sv})
    res = c.post(f'/api/projects/{pid}/calculate').json()
    ranked = sorted(res.values(), key=lambda x: x['rank'])
    print(f'  Project ID={pid}: {name[:45]}')
    for m in ranked:
        print(f'    #{m["rank"]} {m["model_name"]:18s} K={m["k"]:.4f} | {m["label"]}')
    return pid

def main():
    c = httpx.Client(base_url=API, timeout=15)
    try:
        c.get('/api/projects').raise_for_status()
    except Exception:
        print(f"ERROR: server not reachable at {API}")
        print("Start server first: python main.py")
        sys.exit(1)

    print("[1/2] Seeding detection project...")
    seed(c, 'Выбор модели детекции для производства',
         'Сравнение YOLO-моделей для детекции дефектов. Приоритет: скорость + точность на CPU.',
         [{'name':'YOLOv8n','model_type':'detection'},{'name':'YOLOv8s','model_type':'detection'},
          {'name':'YOLOv8m','model_type':'detection'},{'name':'EfficientDet-D0','model_type':'detection'},
          {'name':'YOLOv5su','model_type':'detection'}],
         SCORES_DETECTION)

    print("[2/2] Seeding NLP project...")
    seed(c, 'NLP модели — анализ тональности отзывов',
         'Сравнение моделей сентимент-анализа для русскоязычных отзывов клиентов.',
         [{'name':'rubert-tiny2','model_type':'text'},{'name':'rubert-base','model_type':'text'},
          {'name':'roberta-sentiment','model_type':'text'},{'name':'distilbert-ru','model_type':'text'}],
         SCORES_NLP)

    print(f"\n✓ Ready. Open: http://localhost:8000")
    c.close()

if __name__ == '__main__':
    main()
