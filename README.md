# AI Characher Hub

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Open%20Source-green)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/woka00/ai-characher-hub)](https://github.com/woka00/ai-characher-hub/commits/main)
[![Stars](https://img.shields.io/github/stars/woka00/ai-characher-hub?style=social)](https://github.com/woka00/ai-characher-hub/stargazers)

> 🇬🇧 English below — 🇷🇺 [Русская версия](#русская-версия)

---

## Overview

**AI Characher Hub** is a web platform for objective comparison and benchmarking of AI models — LLMs, NLP pipelines, and computer vision models. It uses weighted evaluation criteria, a composite quality score K ∈ [0..1], and sensitivity analysis to help teams make data-driven model selection decisions.

---

## Screenshots

> **Main dashboard — project overview with model leaderboard**
![Project Dashboard](docs/screenshots/Project_Dashboard.png)

> **Analytics tab — bar, stacked, and radar charts**
![Analytics](docs/screenshots/Analytics.png)

> **K calculation tab — ranked results with per-criteria breakdown**
![K Score](docs/screenshots/K_Score.png)

> **Testing tab — YOLO model benchmarking with progress bar and ETA**
![Testing №1](docs/screenshots/Testing1.png)
![Testing №2](docs/screenshots/Testing2.png)

---

## Features

8 interface tabs, each covering a distinct part of the evaluation workflow:

| Tab | Description |
|-----|-------------|
| 🏠 **Home** | Project overview, use cases, tab guide |
| 📁 **Project** | Dashboard: leader, model list, status |
| ⚖️ **Criteria** | Manage weights and groups, auto-normalize Σw=1 |
| ✏️ **Scores** | Model × criteria scoring matrix; manual or CSV import |
| 🧮 **K Score** | Run calculation, ranked results with breakdown |
| 📐 **Math** | All formulas + step-by-step K calculation per model |
| ⚡ **Testing** | Local YOLO benchmarking with live progress and ETA |
| 📊 **Analytics** | Bar/stacked/radar charts, final report with recommendation |
| 🔬 **Comparison** | Comparison table + sensitivity analysis |

---

## Quick Start

### Windows

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python3 backend/main.py
```

Or double-click `start.bat`.

### Linux / macOS

```bash
python3 -m pip install -r requirements.txt
bash start.sh
```

Opens **http://localhost:8000** with two demo projects (Detection + NLP) preloaded.

> **Python version:** 3.10–3.12 required. Python 3.14 is not yet supported due to pydantic compatibility.

---

## Mathematical Model

```
S_k   = Σ (w_i × a_ik)     — weighted score sum for model k
S_max = 5 × Σ w_i           — maximum possible score
K_k   = S_k / S_max          — quality coefficient [0..1]
```

**K interpretation:**

| Range | Rating |
|-------|--------|
| 0.90 – 1.00 | Excellent — recommended |
| 0.75 – 0.89 | Good |
| 0.60 – 0.74 | Acceptable |
| 0.40 – 0.59 | Weak |
| 0.00 – 0.39 | Not recommended |

All formulas are rendered interactively on the **📐 Math** tab.

---

## Project Structure

```
ai-characher-hub/
├── backend/
│   ├── main.py          # FastAPI app, math model, auto-seed, local testing
│   ├── evaluator.py     # CLI tool for automated model evaluation
│   └── seed_demo.py     # Manual demo data seeding (auto-seed handles this)
├── frontend/
│   └── index.html       # SPA — 8 tabs, no build step required
├── requirements.txt
├── start.bat            # Windows launcher
├── start.sh             # Linux/macOS launcher
└── README.md
```

---

## Adding New Models

### Option 1 — Via UI

1. Open your project → **Project** tab → enter model name (e.g. `YOLOv9c`) and type
2. Click **+ Add**
3. Go to **Scores** tab and fill in the evaluation matrix
4. Go to **K Score** tab → click **Run Calculation**

### Option 2 — Local Automated Testing

Add your model to `_yolo_test_worker` in `backend/main.py`:

```python
weight_file = {
    'YOLOv8n': 'yolov8n.pt',
    # Add your model here:
    'YOLOv9c': 'yolov9c.pt',
}.get(model_name, f'{model_name.lower()}.pt')
```

Requires:
```bash
pip install ultralytics opencv-python
```

### Option 3 — CSV Import

Prepare a CSV:
```
model_name,criterion_name,score
YOLOv9c,Accuracy,4.6
YOLOv9c,Speed,3.8
```

Go to **Scores** tab → click **📂 Import CSV**.

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/projects` | List projects |
| POST | `/api/projects` | Create project |
| GET | `/api/projects/{id}` | Project details |
| DELETE | `/api/projects/{id}` | Delete project |
| POST | `/api/projects/{id}/models` | Add model |
| DELETE | `/api/projects/{id}/models/{mid}` | Remove model |
| GET | `/api/projects/{id}/criteria` | Get criteria |
| POST | `/api/projects/{id}/criteria` | Add criterion |
| PUT | `/api/projects/{id}/criteria/{cid}` | Update criterion |
| DELETE | `/api/projects/{id}/criteria/{cid}` | Delete criterion |
| POST | `/api/projects/{id}/criteria/normalize` | Normalize weights |
| GET | `/api/projects/{id}/scores` | Get scores |
| POST | `/api/projects/{id}/scores` | Submit score |
| POST | `/api/projects/{id}/scores/import` | CSV import |
| POST | `/api/projects/{id}/calculate` | Run K calculation |
| GET | `/api/projects/{id}/results` | Latest results |
| POST | `/api/projects/{id}/sensitivity` | Sensitivity analysis |
| GET | `/api/projects/{id}/report` | Final report |
| **POST** | `/api/projects/{id}/test/start` | **Start local testing** |
| **GET** | `/api/test/{job_id}` | **Polling: test status** |

Full Swagger docs: **http://localhost:8000/docs**

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + SQLite |
| Frontend | HTML/CSS/JS + Chart.js (no build step) |
| ML | ultralytics, torch, transformers (optional) |
| Deploy | Single command, runs on any laptop |


---

---

# Русская версия

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

## Платформа комплексной оценки моделей искусственного интеллекта

Веб-система для объективного сравнения ИИ-моделей по группам критериев с весами, интегральным коэффициентом качества K ∈ [0..1] и анализом чувствительности.

---

## Скриншоты

> **Главный дашборд — обзор проекта и лидерборд моделей**
![Project Dashboard](docs/screenshots/Project_Dashboard.png)

> **Вкладка «Аналитика» — bar, stacked и radar чарты**
![Analytics](docs/screenshots/Analytics.png)

> **Вкладка «Расчёт K» — рейтинг с детализацией по критериям**
![K Score](docs/screenshots/K_Score.png)

> **Вкладка «Тестирование» — прогон YOLO с прогресс-баром и ETA**
![Testing №1](docs/screenshots/Testing1.png)
![Testing №2](docs/screenshots/Testing2.png)


---

## Быстрый старт

### Windows

```bash
python -m pip install -r requirements.txt
python backend/main.py
```

Или двойной клик на `start.bat`.

### Linux / macOS

```bash
# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Запустить
python3 backend/main.py
```

Откроется **http://localhost:8000** с двумя готовыми демо-проектами (Detection + NLP).

> **Версия Python:** требуется 3.10–3.12. Python 3.14 пока не поддерживается из-за несовместимости с pydantic.

---

## Возможности

### 8 вкладок интерфейса

| Вкладка | Описание |
|---------|----------|
| 🏠 **Главная** | Описание проекта, для кого предназначен, обзор всех вкладок |
| 📁 **Проект** | Дашборд: лидер рейтинга, модели, статус |
| ⚖️ **Критерии** | Управление весами и группами критериев, нормировка Σw=1 |
| ✏️ **Оценки** | Матрица оценок модели × критерии. Ввод вручную или импорт CSV |
| 🧮 **Расчёт K** | Запуск расчёта по формулам ТЗ, рейтинг с детализацией |
| 📐 **Математика** | Все формулы с подписями + пошаговый расчёт K для каждой модели |
| ⚡ **Тестирование** | Локальный прогон YOLO с прогресс-баром и ETA |
| 📊 **Аналитика** | Bar/stacked/radar чарты, финальный отчёт с рекомендацией |
| 🔬 **Сравнение** | Сравнительная таблица + анализ чувствительности |

---

## Структура проекта

```
ai-characher-hub/
├── backend/
│   ├── main.py          # FastAPI + математическая модель + auto-seed + локальное тестирование
│   ├── evaluator.py     # CLI-утилита для автоматической оценки моделей
│   └── seed_demo.py     # Ручное наполнение демо-данными (не нужно — есть auto-seed)
├── frontend/
│   └── index.html       # SPA — 8 вкладок, без сборки
├── requirements.txt
├── start.bat            # Запуск на Windows
├── start.sh             # Запуск на Linux/macOS
└── README.md
```

---

## Математическая модель

```
S_k   = Σ (w_i × a_ik)     — взвешенная сумма оценок модели k
S_max = 5 × Σ w_i           — максимально возможная сумма
K_k   = S_k / S_max          — итоговый коэффициент качества [0..1]
```

**Интерпретация K:**

| Диапазон | Оценка |
|----------|--------|
| 0.90 – 1.00 | Отличная модель — рекомендуется |
| 0.75 – 0.89 | Хорошая модель |
| 0.60 – 0.74 | Приемлемая модель |
| 0.40 – 0.59 | Слабая модель |
| 0.00 – 0.39 | Не рекомендуется |

---

## ⚙️ Как внедрять новые модели детекции

### Способ 1 — через UI

1. Открой проект → вкладка «Проект» → введи название (напр. `YOLOv9c`) и тип
2. Нажми «+ Добавить»
3. Перейди на вкладку «Оценки» и заполни матрицу
4. На вкладке «Расчёт K» нажми «Запустить расчёт»

### Способ 2 — автоматическое локальное тестирование

Добавь модель в `_yolo_test_worker` в `backend/main.py`:

```python
weight_file = {
    'YOLOv8n': 'yolov8n.pt',
    # Добавь сюда свою модель:
    'YOLOv9c': 'yolov9c.pt',
}.get(model_name, f'{model_name.lower()}.pt')
```

Требует:
```bash
pip install ultralytics opencv-python
```

### Способ 3 — массовый импорт через CSV

```
model_name,criterion_name,score
YOLOv9c,Точность ответа,4.6
YOLOv9c,Скорость ответа,3.8
```

Вкладка «Оценки» → «📂 Импорт CSV».

---

## Нормировка метрик в шкалу 1–5

```python
# Монотонно возрастающие (больше = лучше)
acc_score = round(avg_conf * 5, 2)

# Монотонно убывающие (меньше = лучше)
size_score = (5.0 if size_mb < 10 else 4.0 if size_mb < 30
              else 3.0 if size_mb < 80 else 2.0)

# Пороговые шкалы (FPS, latency)
speed_score = (5.0 if avg_fps >= 30 else 4.0 if avg_fps >= 15
               else 3.0 if avg_fps >= 5 else 2.0 if avg_fps >= 2 else 1.0)
```

---

## API эндпоинты

| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/api/projects` | Список проектов |
| POST | `/api/projects` | Создать проект |
| GET | `/api/projects/{id}` | Детали проекта |
| DELETE | `/api/projects/{id}` | Удалить |
| POST | `/api/projects/{id}/models` | Добавить модель |
| DELETE | `/api/projects/{id}/models/{mid}` | Удалить модель |
| GET | `/api/projects/{id}/criteria` | Критерии |
| POST | `/api/projects/{id}/criteria` | Добавить критерий |
| PUT | `/api/projects/{id}/criteria/{cid}` | Обновить |
| DELETE | `/api/projects/{id}/criteria/{cid}` | Удалить |
| POST | `/api/projects/{id}/criteria/normalize` | Нормировать веса |
| GET | `/api/projects/{id}/scores` | Получить оценки |
| POST | `/api/projects/{id}/scores` | Выставить оценку |
| POST | `/api/projects/{id}/scores/import` | Импорт CSV |
| POST | `/api/projects/{id}/calculate` | Запустить расчёт K |
| GET | `/api/projects/{id}/results` | Последние результаты |
| POST | `/api/projects/{id}/sensitivity` | Анализ чувствительности |
| GET | `/api/projects/{id}/report` | Финальный отчёт |
| **POST** | `/api/projects/{id}/test/start` | **Запустить локальное тестирование** |
| **GET** | `/api/test/{job_id}` | **Статус тестирования (polling)** |

Swagger: **http://localhost:8000/docs**

---

## Стек

| Слой | Технология |
|------|------------|
| Backend | FastAPI + SQLite |
| Frontend | HTML/CSS/JS + Chart.js (без сборки) |
| ML | ultralytics, torch, transformers (опционально) |
| Запуск | Одна команда, работает на любом ноутбуке |

---
