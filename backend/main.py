"""
═══════════════════════════════════════════════════════════════════════════════
AI CharacherHub — Платформа комплексной оценки моделей искусственного интеллекта
═══════════════════════════════════════════════════════════════════════════════
Backend: FastAPI + SQLite. Запускается одной командой `python main.py`.
При первом запуске автоматически наполняется демо-данными.

Архитектура:
  - main.py             — этот файл, REST API + математическая модель + auto-seed
  - evaluator.py        — автоматический прогон моделей через CLI
  - seed_demo.py        — ручной перезапуск seed (не нужен если используется auto-seed)
  - models/detection/   — папка с весами YOLO (.pt файлы)
  - models/nlp/         — папка-кэш для HuggingFace NLP-моделей

Структура данных в БД (SQLite):
  projects     — проекты оценки (1 проект = 1 задача сравнения моделей)
  ai_models    — модели в рамках проекта
  criteria     — критерии оценки (с весами и группой)
  scores       — оценки модели по критерию (шкала 1–5)
  results      — история расчётов K_k
  test_jobs    — задания на локальное тестирование моделей (новое в v2.0)

Математическая модель (формулы из ТЗ):
  S_k   = Σ(w_i × a_ik)     — взвешенная сумма по модели k
  S_max = 5 × Σ(w_i)         — максимально возможная сумма
  K_k   = S_k / S_max         — итоговый коэффициент качества [0..1]
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Импорты ─────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List
from pathlib import Path
import sqlite3, json, csv, io, os, uuid, time, threading

# ════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ — ПУТИ К ПАПКАМ МОДЕЛЕЙ
# ════════════════════════════════════════════════════════════════════════════
# Эти переменные определяют откуда берутся и куда сохраняются модели.
# При необходимости переопределяй здесь или через переменные окружения.

# Корневая папка backend (рядом с этим файлом)
_BASE_DIR = Path(__file__).parent

# Папка с весами YOLO-моделей (.pt файлы)
# Переопределить: YOLO_MODELS_DIR = Path("/другой/путь/detection")
YOLO_MODELS_DIR = Path(os.getenv("YOLO_MODELS_DIR", str(_BASE_DIR / "models" / "detection")))

# Папка-кэш для HuggingFace NLP-моделей (tokenizer + weights)
# Переопределить: NLP_MODELS_DIR = Path("/другой/путь/nlp")
NLP_MODELS_DIR = Path(os.getenv("NLP_MODELS_DIR", str(_BASE_DIR / "models" / "nlp")))

# Путь к БД SQLite
# Переопределить: DB_PATH = "/другой/путь/ai_eval.db"
DB_PATH = os.getenv("DB_PATH", str(_BASE_DIR / "ai_eval.db"))

# Создаём папки если их нет (при первом запуске)
YOLO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
NLP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Инициализация FastAPI ───────────────────────────────────────────────────
app = FastAPI(title="AI CharacherHub", version="2.0.0", docs_url="/docs")
# CORS разрешён для всех источников — для удобства разработки
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# DB_PATH определён выше в блоке констант


# ════════════════════════════════════════════════════════════════════════════
# СЛОЙ ДАННЫХ — работа с SQLite
# ════════════════════════════════════════════════════════════════════════════

def get_conn():
    """Возвращает соединение с БД с включёнными внешними ключами."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # доступ к колонкам по имени: row['name']
    conn.execute("PRAGMA foreign_keys = ON")  # каскадное удаление
    return conn


def init_db():
    """Создаёт таблицы при первом запуске. Безопасно вызывать многократно."""
    with get_conn() as conn:
        conn.executescript("""
            -- Таблица проектов: одна задача = один проект
            CREATE TABLE IF NOT EXISTS projects (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                description     TEXT    DEFAULT '',
                created_at      TEXT    DEFAULT (datetime('now')),
                last_calculated TEXT,
                report_status   TEXT    DEFAULT 'pending'
            );
            -- Модели в проекте (YOLOv8n, BERT и т.д.)
            CREATE TABLE IF NOT EXISTS ai_models (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name        TEXT    NOT NULL,
                model_type  TEXT    NOT NULL DEFAULT 'custom',
                description TEXT    DEFAULT ''
            );
            -- Критерии оценки с весами (w_i) и группой
            CREATE TABLE IF NOT EXISTS criteria (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name        TEXT    NOT NULL,
                description TEXT    DEFAULT '',
                weight      REAL    NOT NULL DEFAULT 0.1,
                group_name  TEXT    NOT NULL DEFAULT 'accuracy',
                enabled     INTEGER NOT NULL DEFAULT 1
            );
            -- Оценки: одна оценка на пару (модель, критерий), шкала 1–5
            CREATE TABLE IF NOT EXISTS scores (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id     INTEGER NOT NULL REFERENCES ai_models(id)  ON DELETE CASCADE,
                criterion_id INTEGER NOT NULL REFERENCES criteria(id)   ON DELETE CASCADE,
                score        REAL    NOT NULL CHECK(score >= 1 AND score <= 5),
                UNIQUE(model_id, criterion_id)
            );
            -- История расчётов K_k для каждого проекта
            CREATE TABLE IF NOT EXISTS results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                calculated_at TEXT    DEFAULT (datetime('now')),
                result_json   TEXT    NOT NULL
            );
            -- Задания на локальное тестирование (новое в v2.0)
            CREATE TABLE IF NOT EXISTS test_jobs (
                id          TEXT    PRIMARY KEY,        -- UUID
                project_id  INTEGER NOT NULL,
                status      TEXT    NOT NULL,           -- pending/running/done/error
                progress    INTEGER NOT NULL DEFAULT 0, -- 0..100 (%)
                started_at  TEXT    DEFAULT (datetime('now')),
                eta_seconds REAL    DEFAULT 0,          -- осталось секунд
                log         TEXT    DEFAULT '',         -- лог выполнения
                results     TEXT    DEFAULT ''          -- итоговые оценки (JSON)
            );
        """)

init_db()  # инициализация при старте


# ════════════════════════════════════════════════════════════════════════════
# СХЕМЫ ВАЛИДАЦИИ (Pydantic) — защита API от некорректных данных
# ════════════════════════════════════════════════════════════════════════════

# Символы, запрещённые в названиях (защита от XSS и SQL-injection)
SAFE_CHARS = set('<>;\'"\\/')

class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class ProjectCreate(APIModel):
    name: str
    description: str = ""

class ModelCreate(APIModel):
    name: str
    model_type: str = "custom"
    description: str = ""

    @field_validator('name')
    @classmethod
    def safe_name(cls, v):
        if any(c in v for c in SAFE_CHARS):
            raise ValueError('Invalid characters in name')
        return v.strip()

class CriterionCreate(APIModel):
    name: str
    description: str = ""
    weight: float = Field(0.1, ge=0.0, le=1.0)  # вес ограничен [0..1]
    group_name: str = "accuracy"

    @field_validator('name')
    @classmethod
    def safe_name(cls, v):
        if any(c in v for c in SAFE_CHARS):
            raise ValueError('Invalid characters in name')
        return v.strip()

class CriterionUpdate(APIModel):
    name:        Optional[str]   = None
    description: Optional[str]   = None
    weight:      Optional[float] = Field(None, ge=0.0, le=1.0)
    group_name:  Optional[str]   = None
    enabled:     Optional[int]   = None

class ScoreSet(APIModel):
    model_id:     int
    criterion_id: int
    score:        float = Field(..., ge=1.0, le=5.0)  # шкала ТЗ: 1–5

class SensitivityRequest(APIModel):
    criterion_id: int
    delta:        float = Field(0.1, ge=0.01, le=0.5)

class TestStartRequest(APIModel):
    """Запрос на запуск локального тестирования YOLO."""
    model_names: List[str]


# ════════════════════════════════════════════════════════════════════════════
# МАТЕМАТИЧЕСКАЯ МОДЕЛЬ — расчёт коэффициента качества K_k
# ════════════════════════════════════════════════════════════════════════════

def interpret_k(k: float) -> str:
    """Текстовая интерпретация K по 5 градациям из ТЗ."""
    if k >= 0.90: return "Отличная модель — рекомендуется"
    if k >= 0.75: return "Хорошая модель"
    if k >= 0.60: return "Приемлемая модель"
    if k >= 0.40: return "Слабая модель"
    return "Не рекомендуется"


def calculate_k(project_id: int, conn) -> dict:
    """
    ОСНОВНАЯ ФОРМУЛА из ТЗ:
        S_k   = Σ(w_i × a_ik)
        S_max = 5 × Σ(w_i)
        K_k   = S_k / S_max
    Возвращает результаты с детализацией по группам критериев для UI.
    """
    models   = conn.execute("SELECT * FROM ai_models WHERE project_id=?", (project_id,)).fetchall()
    criteria = conn.execute(
        "SELECT * FROM criteria WHERE project_id=? AND enabled=1", (project_id,)
    ).fetchall()
    if not models or not criteria:
        return {}

    results = {}
    for model in models:
        s_k = 0.0          # числитель — взвешенная сумма
        s_max = 0.0        # знаменатель — макс. возможная сумма
        group_details = {}

        # Считаем отдельно по каждой группе для красивой визуализации
        groups = set(c['group_name'] for c in criteria)
        for g in groups:
            g_crit = [c for c in criteria if c['group_name'] == g]
            g_s_k  = 0.0
            g_s_max = 5 * sum(c['weight'] for c in g_crit)
            crit_rows = []

            for c in g_crit:
                row = conn.execute(
                    "SELECT score FROM scores WHERE model_id=? AND criterion_id=?",
                    (model['id'], c['id'])
                ).fetchone()
                score        = row['score'] if row else None
                # Вклад критерия: w_i × a_ik
                contribution = (c['weight'] * score) if score is not None else 0.0
                g_s_k  += contribution
                s_k    += contribution
                s_max  += 5 * c['weight']
                crit_rows.append({
                    "criterion_id":   c['id'],
                    "criterion_name": c['name'],
                    "weight":         c['weight'],
                    "score":          score,
                    "contribution":   round(contribution, 4),
                    "contribution_pct": 0,
                })

            # K для группы — для радарной диаграммы
            group_details[g] = {
                "k":      round(g_s_k / g_s_max, 4) if g_s_max > 0 else 0.0,
                "s_k":    round(g_s_k, 4),
                "s_max":  round(g_s_max, 4),
                "criteria": crit_rows,
            }

        k = round(s_k / s_max, 4) if s_max > 0 else 0.0

        # Процент вклада критериев в общий S_k (для отчёта)
        for g_data in group_details.values():
            for cd in g_data['criteria']:
                cd['contribution_pct'] = round(
                    (cd['contribution'] / s_k * 100) if s_k > 0 else 0, 1
                )

        results[str(model['id'])] = {
            "model_id":   model['id'],
            "model_name": model['name'],
            "model_type": model['model_type'],
            "k":     k,
            "s_k":   round(s_k, 4),
            "s_max": round(s_max, 4),
            "label": interpret_k(k),
            "groups": group_details,
            "rank":  0,
        }

    # Ранжируем по убыванию K
    sorted_ids = sorted(results.keys(), key=lambda mid: -results[mid]['k'])
    for rank, mid in enumerate(sorted_ids, 1):
        results[mid]['rank'] = rank
    return results


def run_sensitivity(project_id: int, criterion_id: int, delta: float, conn) -> dict:
    """
    АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ (обязателен по ТЗ).
    Временно увеличиваем вес критерия, пересчитываем K, смотрим смену лидера.
    После расчёта ОБЯЗАТЕЛЬНО откатываем изменение.
    """
    baseline = calculate_k(project_id, conn)
    if not baseline: return {}

    bl_leader = min(baseline.keys(), key=lambda m: baseline[m]['rank'])
    orig = conn.execute("SELECT weight FROM criteria WHERE id=?", (criterion_id,)).fetchone()
    if not orig: return {}

    orig_w = orig['weight']
    new_w  = min(1.0, orig_w + delta)
    conn.execute("UPDATE criteria SET weight=? WHERE id=?", (new_w, criterion_id))
    modified = calculate_k(project_id, conn)
    # ВАЖНО: возвращаем исходный вес
    conn.execute("UPDATE criteria SET weight=? WHERE id=?", (orig_w, criterion_id))

    new_leader = min(modified.keys(), key=lambda m: modified[m]['rank']) if modified else None
    delta_k = {
        mid: {
            "model_name":  baseline[mid]['model_name'],
            "baseline_k":  baseline[mid]['k'],
            "new_k":       modified.get(mid, {}).get('k', 0),
            "delta":       round(modified.get(mid, {}).get('k', 0) - baseline[mid]['k'], 4),
        }
        for mid in baseline
    }
    return {
        "criterion_id":   criterion_id,
        "delta":          delta,
        "leader_changed": new_leader != bl_leader,
        "baseline_leader": baseline[bl_leader]['model_name'],
        "new_leader":     modified[new_leader]['model_name'] if new_leader else None,
        "models":         delta_k,
    }


def validate_project_exists(pid: int, conn):
    """Хелпер: бросает 404 если проекта нет."""
    if not conn.execute("SELECT 1 FROM projects WHERE id=?", (pid,)).fetchone():
        raise HTTPException(404, f"Project {pid} not found")


# ════════════════════════════════════════════════════════════════════════════
# API РОУТЫ — Проекты
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/projects")
def list_projects():
    """Список проектов с краткой статистикой для бокового меню."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY id DESC").fetchall()
        out = []
        for r in rows:
            p = dict(r)
            p['model_count']     = conn.execute("SELECT COUNT(*) FROM ai_models WHERE project_id=?", (r['id'],)).fetchone()[0]
            p['criterion_count'] = conn.execute("SELECT COUNT(*) FROM criteria WHERE project_id=? AND enabled=1", (r['id'],)).fetchone()[0]
            latest = conn.execute("SELECT result_json FROM results WHERE project_id=? ORDER BY id DESC LIMIT 1", (r['id'],)).fetchone()
            if latest:
                res    = json.loads(latest['result_json'])
                leader = next((v for v in res.values() if v['rank'] == 1), None)
                p['leader']   = leader['model_name'] if leader else None
                p['leader_k'] = leader['k']          if leader else None
            else:
                p['leader'] = p['leader_k'] = None
            out.append(p)
        return out


@app.post("/api/projects", status_code=201)
def create_project(data: ProjectCreate):
    """Создать проект + сразу завести 10 стандартных критериев."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO projects (name, description) VALUES (?,?)",
            (data.name.strip(), data.description)
        )
        pid = cur.lastrowid
        defaults = [
            ("Точность ответа",        "Насколько точно модель решает задачу",                0.30, "accuracy"),
            ("Глубина и полнота",       "Охватывает ли все аспекты задачи",                   0.20, "accuracy"),
            ("Логичность и структура",  "Структурированность и последовательность вывода",     0.15, "accuracy"),
            ("Гибкость интерпретации",  "Работа с неоднозначными входными данными",            0.15, "accuracy"),
            ("Устойчивость к шуму",     "Стабильность при зашумлённых входных данных",         0.20, "robustness"),
            ("Обработка сложных задач", "Работа с многоэтапными и составными запросами",       0.15, "robustness"),
            ("Скорость ответа",         "Время инференса на CPU",                              0.10, "robustness"),
            ("Контекстная согласованность","Сохранение связи с контекстом задачи",             0.15, "context"),
            ("Адаптивность",            "Подстройка под специфику конкретной задачи",          0.10, "context"),
            ("Компактность модели",     "Размер модели и требования к памяти",                 0.10, "context"),
        ]
        for name, desc, w, grp in defaults:
            conn.execute(
                "INSERT INTO criteria (project_id,name,description,weight,group_name) VALUES (?,?,?,?,?)",
                (pid, name, desc, w, grp)
            )
        return {"id": pid, "name": data.name}


@app.get("/api/projects/{pid}")
def get_project(pid: int):
    """Полная инфа о проекте: модели + критерии + мета."""
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        p = dict(conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone())
        p['models']   = [dict(m) for m in conn.execute("SELECT * FROM ai_models WHERE project_id=?", (pid,)).fetchall()]
        p['criteria'] = [dict(c) for c in conn.execute("SELECT * FROM criteria WHERE project_id=? ORDER BY group_name,id", (pid,)).fetchall()]
        return p


@app.delete("/api/projects/{pid}")
def delete_project(pid: int):
    """Каскадно удалить проект."""
    with get_conn() as conn:
        conn.execute("DELETE FROM projects WHERE id=?", (pid,))
    return {"ok": True}


# ── Модели ───────────────────────────────────────────────────────────────────
@app.get("/api/projects/{pid}/models")
def list_models(pid: int):
    with get_conn() as conn:
        return [dict(m) for m in conn.execute("SELECT * FROM ai_models WHERE project_id=?", (pid,)).fetchall()]

@app.post("/api/projects/{pid}/models", status_code=201)
def add_model(pid: int, data: ModelCreate):
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        cur = conn.execute(
            "INSERT INTO ai_models (project_id,name,model_type,description) VALUES (?,?,?,?)",
            (pid, data.name, data.model_type, data.description)
        )
        return {"id": cur.lastrowid, "name": data.name}

@app.delete("/api/projects/{pid}/models/{mid}")
def delete_model(pid: int, mid: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM ai_models WHERE id=? AND project_id=?", (mid, pid))
    return {"ok": True}


# ── Критерии ─────────────────────────────────────────────────────────────────
@app.get("/api/projects/{pid}/criteria")
def list_criteria(pid: int):
    with get_conn() as conn:
        return [dict(c) for c in conn.execute(
            "SELECT * FROM criteria WHERE project_id=? ORDER BY group_name,id", (pid,)
        ).fetchall()]

@app.post("/api/projects/{pid}/criteria", status_code=201)
def add_criterion(pid: int, data: CriterionCreate):
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        cur = conn.execute(
            "INSERT INTO criteria (project_id,name,description,weight,group_name) VALUES (?,?,?,?,?)",
            (pid, data.name, data.description, data.weight, data.group_name)
        )
        return {"id": cur.lastrowid}

@app.put("/api/projects/{pid}/criteria/{cid}")
def update_criterion(pid: int, cid: int, data: CriterionUpdate):
    """Частичное обновление — только переданные поля."""
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "Nothing to update")
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE criteria SET {set_clause} WHERE id=? AND project_id=?",
            list(updates.values()) + [cid, pid]
        )
    return {"ok": True}

@app.delete("/api/projects/{pid}/criteria/{cid}")
def delete_criterion(pid: int, cid: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM criteria WHERE id=? AND project_id=?", (cid, pid))
    return {"ok": True}

@app.post("/api/projects/{pid}/criteria/normalize")
def normalize_weights(pid: int):
    """Нормировка весов внутри каждой группы до суммы 1.0."""
    with get_conn() as conn:
        crit = conn.execute("SELECT * FROM criteria WHERE project_id=? AND enabled=1", (pid,)).fetchall()
        groups = {}
        for c in crit:
            groups.setdefault(c['group_name'], []).append(c)
        for g_list in groups.values():
            total = sum(c['weight'] for c in g_list)
            if total > 0:
                for c in g_list:
                    conn.execute("UPDATE criteria SET weight=? WHERE id=?",
                                 (round(c['weight']/total, 4), c['id']))
    return {"ok": True}


# ── Оценки ───────────────────────────────────────────────────────────────────
@app.get("/api/projects/{pid}/scores")
def get_scores(pid: int):
    """Все оценки с именами моделей и критериев."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s.model_id, s.criterion_id, s.score,
                   m.name as model_name, c.name as criterion_name, c.group_name
            FROM scores s
            JOIN ai_models m ON m.id=s.model_id
            JOIN criteria  c ON c.id=s.criterion_id
            WHERE m.project_id=?
        """, (pid,)).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/projects/{pid}/scores")
def set_score(pid: int, data: ScoreSet):
    """UPSERT оценки (вставка или обновление)."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO scores (model_id,criterion_id,score) VALUES (?,?,?)
            ON CONFLICT(model_id,criterion_id) DO UPDATE SET score=excluded.score
        """, (data.model_id, data.criterion_id, data.score))
    return {"ok": True}


@app.post("/api/projects/{pid}/scores/import")
async def import_csv(pid: int, file: UploadFile = File(...)):
    """
    Импорт оценок из CSV.
    Формат: model_name,criterion_name,score
    Защита: максимум 2MB, проверка диапазона score [1..5].
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, "Only CSV files allowed")
    content = await file.read()
    if len(content) > 2 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 2MB)")

    text   = content.decode('utf-8-sig')  # поддержка BOM от Excel
    reader = csv.DictReader(io.StringIO(text))
    required = {'model_name', 'criterion_name', 'score'}
    if not required.issubset(set(reader.fieldnames or [])):
        raise HTTPException(400, f"CSV must have columns: {required}")

    imported, errors = 0, []
    with get_conn() as conn:
        for i, row in enumerate(reader, 2):
            try:
                score = float(row['score'])
                if not 1 <= score <= 5:
                    errors.append(f"Row {i}: score {score} out of range"); continue
                model = conn.execute(
                    "SELECT id FROM ai_models WHERE project_id=? AND name=?",
                    (pid, row['model_name'].strip())
                ).fetchone()
                crit = conn.execute(
                    "SELECT id FROM criteria WHERE project_id=? AND name=?",
                    (pid, row['criterion_name'].strip())
                ).fetchone()
                if not model: errors.append(f"Row {i}: model not found"); continue
                if not crit:  errors.append(f"Row {i}: criterion not found"); continue
                conn.execute("""
                    INSERT INTO scores (model_id,criterion_id,score) VALUES (?,?,?)
                    ON CONFLICT(model_id,criterion_id) DO UPDATE SET score=excluded.score
                """, (model['id'], crit['id'], score))
                imported += 1
            except (ValueError, KeyError) as e:
                errors.append(f"Row {i}: {e}")
    return {"imported": imported, "errors": errors}


# ── Расчёт и результаты ──────────────────────────────────────────────────────
@app.post("/api/projects/{pid}/calculate")
def run_calculation(pid: int):
    """Запустить расчёт K_k и сохранить в историю."""
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        results = calculate_k(pid, conn)
        if not results:
            raise HTTPException(400, "No models or criteria.")
        conn.execute("INSERT INTO results (project_id,result_json) VALUES (?,?)",
                     (pid, json.dumps(results)))
        conn.execute("UPDATE projects SET last_calculated=datetime('now'), report_status='ready' WHERE id=?",
                     (pid,))
        return results

@app.get("/api/projects/{pid}/results")
def get_results(pid: int):
    """Последний расчёт по проекту."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM results WHERE project_id=? ORDER BY id DESC LIMIT 1", (pid,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "No results yet.")
        return {"calculated_at": row['calculated_at'], "results": json.loads(row['result_json'])}

@app.post("/api/projects/{pid}/sensitivity")
def sensitivity(pid: int, data: SensitivityRequest):
    with get_conn() as conn:
        result = run_sensitivity(pid, data.criterion_id, data.delta, conn)
        if not result:
            raise HTTPException(400, "Could not run analysis.")
        return result


# ── Отчёт ────────────────────────────────────────────────────────────────────
@app.get("/api/projects/{pid}/report")
def get_report(pid: int):
    """Финальный отчёт с текстовой рекомендацией по результатам."""
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        latest = conn.execute(
            "SELECT * FROM results WHERE project_id=? ORDER BY id DESC LIMIT 1", (pid,)
        ).fetchone()
        if not latest:
            raise HTTPException(404, "No results.")
        proj    = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        results = json.loads(latest['result_json'])
        sorted_models = sorted(results.values(), key=lambda x: x['rank'])
        winner  = sorted_models[0]

        # Топ-3 сильных критериев у победителя
        all_contrib = []
        for g_data in winner['groups'].values():
            for cd in g_data['criteria']:
                if cd['score'] is not None:
                    all_contrib.append(cd)
        all_contrib.sort(key=lambda x: -x['contribution'])

        k = winner['k']
        gap = round(sorted_models[0]['k'] - sorted_models[1]['k'], 4) if len(sorted_models) > 1 else 0
        if k >= 0.90: v = f"Модель {winner['model_name']} показала отличные результаты (K={k}) и рекомендуется к применению."
        elif k >= 0.75: v = f"Модель {winner['model_name']} показала хорошие результаты (K={k}) и подходит для большинства задач."
        elif k >= 0.60: v = f"Модель {winner['model_name']} показала приемлемые результаты (K={k}), требуется доработка."
        else:           v = f"Модель {winner['model_name']} лидирует (K={k}), но ни одна не показала достаточного качества."
        if gap < 0.05: v += " Разрыв с конкурентом минимален — лидер может смениться при изменении весов."
        else:          v += f" Отрыв от второго места: {gap} — выбор устойчив."

        return {
            "project_name":    proj['name'],
            "calculated_at":   latest['calculated_at'],
            "winner":          {"name": winner['model_name'], "k": k, "label": winner['label']},
            "ranking":         [{"rank": m['rank'], "name": m['model_name'], "k": m['k'], "label": m['label']} for m in sorted_models],
            "winner_strengths": all_contrib[:3],
            "recommendation":  v,
        }


# ════════════════════════════════════════════════════════════════════════════
# ЛОКАЛЬНОЕ ТЕСТИРОВАНИЕ МОДЕЛЕЙ (новое в v2.0)
# ════════════════════════════════════════════════════════════════════════════
# Запускает реальные модели YOLO через ultralytics и автоматически выставляет
# оценки. Работает в фоновом потоке, фронтенд опрашивает статус через polling.
# ════════════════════════════════════════════════════════════════════════════

def _update_job(job_id: str, **fields):
    """Обновить поля задания тестирования."""
    if not fields: return
    set_clause = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE test_jobs SET {set_clause} WHERE id=?",
                     list(fields.values()) + [job_id])


def _yolo_test_worker(job_id: str, project_id: int, model_names: list):
    """
    Фоновый поток для прогона YOLO-моделей.
    
    Алгоритм:
    1. Для каждой модели:
       - Загружаем веса (скачиваются автоматически с GitHub если их нет)
       - Прогоняем на 5 тестовых картинках (встроенные в ultralytics)
       - На каждой картинке: чистый прогон + прогон с гауссовым шумом
       - Собираем метрики: confidence, FPS, размер модели, устойчивость к шуму
    2. Нормируем сырые метрики в оценки 1–5
    3. Сохраняем оценки в БД через UPSERT
    
    Прогресс обновляется после каждого изображения,
    ETA считается линейной экстраполяцией прошедшего времени.
    """
    try:
        # Импорт ленивый — если ultralytics не стоит, выдаём понятную ошибку
        try:
            from ultralytics import YOLO
            from ultralytics.utils import ASSETS
            import numpy as np
            import cv2
        except ImportError as e:
            _update_job(job_id, status='error',
                        log=f'Ошибка импорта: {e}\nУстановите: pip install ultralytics opencv-python numpy')
            return

        # Встроенные тестовые изображения (zidane.jpg, bus.jpg и т.д.)
        test_images = list(ASSETS.glob("*.jpg"))[:5]
        if not test_images:
            _update_job(job_id, status='error', log='Не найдены тестовые изображения')
            return

        log_lines = [f'Начало тестирования: {len(model_names)} моделей × {len(test_images)} изображений']
        # Каждая картинка проходит 2 этапа: чистый + зашумлённый
        total_steps = len(model_names) * len(test_images) * 2
        current_step = 0
        start_time = time.time()
        results = {}

        for model_name in model_names:
            try:
                log_lines.append(f'\n[{model_name}] Загрузка модели...')
                _update_job(job_id, log='\n'.join(log_lines), status='running')

                # Сопоставляем имя модели → файл весов (.pt)
                # Модели хранятся в YOLO_MODELS_DIR (backend/models/detection/)
                # При первом обращении ultralytics сам скачает веса с GitHub
                # и положит их в текущую директорию; мы принудительно
                # указываем нашу папку через os.chdir перед инициализацией.
                weight_map = {
                    'YOLOv8n': 'yolov8n.pt',
                    'YOLOv8s': 'yolov8s.pt',
                    'YOLOv8m': 'yolov8m.pt',
                    'YOLOv8l': 'yolov8l.pt',
                    'YOLOv8x': 'yolov8x.pt',
                    # YOLOv9 — добавь сюда: 'YOLOv9c': 'yolov9c.pt'
                    # YOLOv10 — добавь сюда: 'YOLOv10n': 'yolov10n.pt'
                }
                weight_filename = weight_map.get(model_name, f'{model_name.lower()}.pt')
                # Полный путь до файла весов в нашей папке
                weight_file = str(YOLO_MODELS_DIR / weight_filename)
                # Меняем рабочую директорию чтобы ultralytics скачал
                # модель именно в нашу папку YOLO_MODELS_DIR
                _prev_dir = os.getcwd()
                os.chdir(str(YOLO_MODELS_DIR))

                model = YOLO(weight_file)
                times, confs = [], []
                base_detections, noise_detections = [], []

                for img_path in test_images:
                    img = cv2.imread(str(img_path))
                    if img is None: continue

                    # ── Этап 1: чистый прогон → точность и скорость ──
                    t0 = time.time()
                    r = model(img, verbose=False)[0]
                    times.append(time.time() - t0)
                    if len(r.boxes):
                        confs.extend(r.boxes.conf.tolist())
                    base_detections.append(len(r.boxes))

                    current_step += 1
                    elapsed = time.time() - start_time
                    progress = int(current_step / total_steps * 100)
                    # ETA = (среднее время на шаг) × (осталось шагов)
                    eta = (elapsed / current_step * (total_steps - current_step)) if current_step > 0 else 0
                    _update_job(job_id, progress=progress, eta_seconds=round(eta, 1))

                    # ── Этап 2: прогон с гауссовым шумом → устойчивость ──
                    noise = np.clip(img.astype(np.int16) + np.random.normal(0, 30, img.shape), 0, 255).astype(np.uint8)
                    r_noise = model(noise, verbose=False)[0]
                    noise_detections.append(len(r_noise.boxes))

                    current_step += 1
                    progress = int(current_step / total_steps * 100)
                    eta = (elapsed / current_step * (total_steps - current_step)) if current_step > 0 else 0
                    _update_job(job_id, progress=progress, eta_seconds=round(eta, 1))

                # ── Агрегированные метрики ──
                avg_conf = sum(confs) / len(confs) if confs else 0
                avg_fps  = 1 / (sum(times) / len(times)) if times else 0
                avg_base = sum(base_detections) / max(len(base_detections), 1)
                avg_noise = sum(noise_detections) / max(len(noise_detections), 1)

                size_mb = 0
                try:
                    size_mb = os.path.getsize(weight_file) / 1e6
                except: pass
                finally:
                    # Возвращаем рабочую директорию обратно
                    os.chdir(_prev_dir)

                # ── НОРМИРОВКА метрик в шкалу 1–5 ──
                # FPS → пороговая шкала
                speed_score = (5.0 if avg_fps >= 30 else 4.0 if avg_fps >= 15
                               else 3.0 if avg_fps >= 5 else 2.0 if avg_fps >= 2 else 1.0)
                # Confidence в [0..1] → умножаем на 5
                acc_score = round(avg_conf * 5, 2)
                # Устойчивость = ratio детекций при шуме / без шума
                noise_ratio = avg_noise / max(avg_base, 1)
                robust_score = (5.0 if noise_ratio >= 0.9 else 4.0 if noise_ratio >= 0.75
                                else 3.0 if noise_ratio >= 0.55 else 2.0 if noise_ratio >= 0.35 else 1.0)
                # Размер модели → меньше = лучше
                size_score = (5.0 if size_mb < 10 else 4.0 if size_mb < 30
                              else 3.0 if size_mb < 80 else 2.0)

                model_scores = {
                    "Точность ответа":              acc_score,
                    "Глубина и полнота":            round(min(avg_base / 3 * 5, 5), 2),
                    "Логичность и структура":       4.0,  # детекция = структурированный вывод
                    "Гибкость интерпретации":       3.5,
                    "Устойчивость к шуму":          robust_score,
                    "Обработка сложных задач":      round(acc_score * 0.9, 2),
                    "Скорость ответа":              speed_score,
                    "Контекстная согласованность":  3.5,
                    "Адаптивность":                 3.0,
                    "Компактность модели":          size_score,
                }
                results[model_name] = model_scores

                log_lines.append(f'  ✓ {model_name}: conf={avg_conf:.3f} fps={avg_fps:.1f} size={size_mb:.1f}MB')
                _update_job(job_id, log='\n'.join(log_lines))

            except Exception as e:
                log_lines.append(f'  ✗ Ошибка {model_name}: {e}')
                _update_job(job_id, log='\n'.join(log_lines))

        # ── Записываем оценки в БД ──
        with get_conn() as conn:
            crit_map = {x['name']: x['id'] for x in conn.execute(
                "SELECT id,name FROM criteria WHERE project_id=?", (project_id,)).fetchall()}
            mod_map = {x['name']: x['id'] for x in conn.execute(
                "SELECT id,name FROM ai_models WHERE project_id=?", (project_id,)).fetchall()}
            for mn, scores in results.items():
                mid = mod_map.get(mn)
                if not mid: continue
                for cn, sv in scores.items():
                    cid = crit_map.get(cn)
                    if cid:
                        conn.execute(
                            "INSERT INTO scores (model_id,criterion_id,score) VALUES (?,?,?) "
                            "ON CONFLICT(model_id,criterion_id) DO UPDATE SET score=excluded.score",
                            (mid, cid, sv))

        log_lines.append(f'\n✓ Готово. Оценки записаны в проект.')
        _update_job(job_id, status='done', progress=100, eta_seconds=0,
                    log='\n'.join(log_lines), results=json.dumps(results))

    except Exception as e:
        # Любые непойманные ошибки логируем
        import traceback
        _update_job(job_id, status='error',
                    log=f'Ошибка: {e}\n{traceback.format_exc()}')


@app.post("/api/projects/{pid}/test/start")
def start_test(pid: int, req: TestStartRequest):
    """
    Запустить локальное тестирование YOLO-моделей детекции.
    Возвращает job_id — фронтенд опрашивает /api/test/{job_id} для прогресса.
    """
    job_id = str(uuid.uuid4())
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        conn.execute(
            "INSERT INTO test_jobs (id,project_id,status,progress) VALUES (?,?,?,?)",
            (job_id, pid, 'pending', 0)
        )
    # Запускаем тестирование в фоновом потоке (не блокирует API)
    thread = threading.Thread(
        target=_yolo_test_worker,
        args=(job_id, pid, req.model_names),
        daemon=True  # daemon = поток умрёт вместе с сервером
    )
    thread.start()
    return {"job_id": job_id, "status": "started"}


@app.get("/api/test/{job_id}")
def get_test_status(job_id: str):
    """Опрос статуса задания. Фронтенд вызывает каждые 500мс для прогресс-бара."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM test_jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job not found")
        return dict(row)


# ════════════════════════════════════════════════════════════════════════════
# ЛОКАЛЬНОЕ ТЕСТИРОВАНИЕ NLP-МОДЕЛЕЙ (новое в v2.1)
# ════════════════════════════════════════════════════════════════════════════
# Запускает HuggingFace sentiment-analysis модели на наборе тестовых текстов,
# собирает метрики (accuracy, speed, robustness) и записывает оценки в проект.
# Модели кэшируются в NLP_MODELS_DIR (backend/models/nlp/).
# ════════════════════════════════════════════════════════════════════════════

# Тестовые тексты для оценки NLP-моделей
# Каждый текст имеет ожидаемую тональность: pos/neg/neu
NLP_TEST_TEXTS = [
    ("Этот продукт просто отличный, очень доволен покупкой!", "pos"),
    ("Ужасное качество, полное разочарование и трата денег.", "neg"),
    ("Нормально, ничего особенного, ожидал большего.", "neu"),
    ("Рекомендую всем! Превзошло все мои ожидания!", "pos"),
    ("Брак, сломалось через неделю, не советую никому.", "neg"),
    # Тексты с шумом — опечатки, сленг (для теста устойчивости)
    ("оч крутая штука, всем советую взять!!!", "pos"),
    ("хрень полная, выбросил сразу", "neg"),
    ("Ну так, норм в принципе если не придираться", "neu"),
]

# Маппинг HuggingFace label → наша категория pos/neg/neu
_LABEL_MAP = {
    'POSITIVE': 'pos', 'NEGATIVE': 'neg', 'NEUTRAL': 'neu',
    'LABEL_0': 'neg', 'LABEL_1': 'neu', 'LABEL_2': 'pos',  # 3-class
    'positive': 'pos', 'negative': 'neg', 'neutral': 'neu',
}

# Поддерживаемые NLP-модели → HuggingFace model id
NLP_MODEL_REGISTRY = {
    'rubert-tiny2':        'cointegrated/rubert-tiny2',
    'rubert-base':         'blanchefort/rubert-base-cased-sentiment',
    'roberta-sentiment':   'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'distilbert-ru':       'Tatyana/distilbert-base-multilingual-cased-sentiments-student',
    # Чтобы добавить новую модель — просто добавь строку:
    # 'мой-ярлык': 'huggingface/model-id',
}


def _nlp_test_worker(job_id: str, project_id: int, model_names: list):
    """
    Фоновый поток для тестирования NLP-моделей.

    Алгоритм:
    1. Для каждой модели загружаем pipeline через HuggingFace transformers.
       Кэш — NLP_MODELS_DIR (backend/models/nlp/), не засоряет системный кэш.
    2. Прогоняем NLP_TEST_TEXTS: чистые тексты + зашумлённые (опечатки/caps).
    3. Считаем: accuracy (верных предсказаний), confidence, скорость, размер.
    4. Нормируем в шкалу 1–5 и записываем через UPSERT в таблицу scores.
    """
    try:
        try:
            from transformers import pipeline
            import torch
        except ImportError as e:
            _update_job(job_id, status='error',
                        log=f'Ошибка импорта: {e}\nУстановите: pip install transformers torch')
            return

        log_lines = [f'Начало NLP-тестирования: {len(model_names)} моделей × {len(NLP_TEST_TEXTS)} текстов']
        total_steps = len(model_names) * len(NLP_TEST_TEXTS)
        current_step = 0
        start_time = time.time()
        results = {}

        for model_name in model_names:
            # Получаем HuggingFace model id из реестра
            hf_model_id = NLP_MODEL_REGISTRY.get(model_name)
            if not hf_model_id:
                log_lines.append(f'  ✗ {model_name}: не найден в NLP_MODEL_REGISTRY')
                _update_job(job_id, log='\n'.join(log_lines))
                continue

            try:
                log_lines.append(f'\n[{model_name}] Загрузка {hf_model_id}...')
                _update_job(job_id, log='\n'.join(log_lines), status='running')

                # Указываем кэш в нашу папку NLP_MODELS_DIR
                pipe = pipeline(
                    'sentiment-analysis',
                    model=hf_model_id,
                    truncation=True,
                    cache_dir=str(NLP_MODELS_DIR),
                )

                times, confs, correct, robust_correct = [], [], 0, 0

                for text, expected_label in NLP_TEST_TEXTS:
                    # ── Чистый прогон ──
                    t0 = time.time()
                    out = pipe(text)[0]
                    times.append(time.time() - t0)
                    confs.append(out['score'])
                    predicted = _LABEL_MAP.get(out['label'].upper(), 'neu')
                    if predicted == expected_label:
                        correct += 1

                    # ── Зашумлённый текст: ALL CAPS ──
                    noisy = text.upper()
                    out_n = pipe(noisy)[0]
                    pred_n = _LABEL_MAP.get(out_n['label'].upper(), 'neu')
                    if pred_n == expected_label:
                        robust_correct += 1

                    current_step += 1
                    elapsed = time.time() - start_time
                    progress = int(current_step / total_steps * 100)
                    eta = (elapsed / current_step * (total_steps - current_step)) if current_step > 0 else 0
                    _update_job(job_id, progress=progress, eta_seconds=round(eta, 1))

                # ── Агрегированные метрики ──
                n = len(NLP_TEST_TEXTS)
                accuracy   = correct / n         # доля верных ответов [0..1]
                robustness = robust_correct / n  # то же на зашумлённых
                avg_conf   = sum(confs) / len(confs)
                avg_ms     = sum(times) / len(times) * 1000  # мс на текст

                # Размер модели через кол-во параметров
                try:
                    params_m = sum(p.numel() for p in pipe.model.parameters()) / 1e6
                except:
                    params_m = 100  # fallback

                # ── Нормировка в шкалу 1–5 ──
                acc_score    = round(accuracy * 5, 2)      # 100% → 5.0
                conf_score   = round(avg_conf * 5, 2)
                robust_score = round(robustness * 5, 2)
                # Скорость: <50мс=5, <150мс=4, <400мс=3, <800мс=2, >800мс=1
                speed_score  = (5.0 if avg_ms < 50 else 4.0 if avg_ms < 150
                                else 3.0 if avg_ms < 400 else 2.0 if avg_ms < 800 else 1.0)
                # Размер: <30M=5, <80M=4, <150M=3, <300M=2, >300M=1
                size_score   = (5.0 if params_m < 30 else 4.0 if params_m < 80
                                else 3.0 if params_m < 150 else 2.0 if params_m < 300 else 1.0)

                model_scores = {
                    "Точность ответа":              acc_score,
                    "Глубина и полнота":            round((acc_score + conf_score) / 2, 2),
                    "Логичность и структура":       conf_score,
                    "Гибкость интерпретации":       round(robust_score * 0.9, 2),
                    "Устойчивость к шуму":          robust_score,
                    "Обработка сложных задач":      round((acc_score + robust_score) / 2, 2),
                    "Скорость ответа":              speed_score,
                    "Контекстная согласованность":  acc_score,
                    "Адаптивность":                 round(robust_score * 0.85, 2),
                    "Компактность модели":          size_score,
                }
                results[model_name] = model_scores

                log_lines.append(
                    f'  ✓ {model_name}: accuracy={accuracy:.1%} robustness={robustness:.1%} '
                    f'speed={avg_ms:.0f}мс params={params_m:.0f}M'
                )
                _update_job(job_id, log='\n'.join(log_lines))

            except Exception as e:
                log_lines.append(f'  ✗ Ошибка {model_name}: {e}')
                _update_job(job_id, log='\n'.join(log_lines))

        # ── Записываем оценки в БД ──
        with get_conn() as conn:
            crit_map = {x['name']: x['id'] for x in conn.execute(
                "SELECT id,name FROM criteria WHERE project_id=?", (project_id,)).fetchall()}
            mod_map = {x['name']: x['id'] for x in conn.execute(
                "SELECT id,name FROM ai_models WHERE project_id=?", (project_id,)).fetchall()}
            for mn, scores in results.items():
                mid = mod_map.get(mn)
                if not mid: continue
                for cn, sv in scores.items():
                    cid = crit_map.get(cn)
                    if cid:
                        conn.execute(
                            "INSERT INTO scores (model_id,criterion_id,score) VALUES (?,?,?) "
                            "ON CONFLICT(model_id,criterion_id) DO UPDATE SET score=excluded.score",
                            (mid, cid, sv))

        log_lines.append(f'\n✓ Готово. NLP-оценки записаны в проект.')
        _update_job(job_id, status='done', progress=100, eta_seconds=0,
                    log='\n'.join(log_lines), results=json.dumps(results))

    except Exception as e:
        import traceback
        _update_job(job_id, status='error',
                    log=f'Ошибка: {e}\n{traceback.format_exc()}')


@app.post("/api/projects/{pid}/test/nlp/start")
def start_nlp_test(pid: int, req: TestStartRequest):
    """
    Запустить локальное тестирование NLP-моделей.
    Модели кэшируются в backend/models/nlp/.
    Возвращает job_id — фронтенд опрашивает /api/test/{job_id} для прогресса.
    """
    job_id = str(uuid.uuid4())
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        conn.execute(
            "INSERT INTO test_jobs (id,project_id,status,progress) VALUES (?,?,?,?)",
            (job_id, pid, 'pending', 0)
        )
    thread = threading.Thread(
        target=_nlp_test_worker,
        args=(job_id, pid, req.model_names),
        daemon=True
    )
    thread.start()
    return {"job_id": job_id, "status": "started"}


# ════════════════════════════════════════════════════════════════════════════
# AUTO-SEED — наполнение демо-данными при первом запуске
# ════════════════════════════════════════════════════════════════════════════
# Реалистичные оценки на основе публичных бенчмарков COCO/HuggingFace
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


def _seed_project(conn, name, desc, models_cfg, scores_data):
    """Создать проект + модели + критерии + оценки + посчитать K."""
    cur = conn.execute("INSERT INTO projects (name,description) VALUES (?,?)", (name, desc))
    pid = cur.lastrowid
    defaults = [
        ("Точность ответа","Насколько точно модель решает задачу",0.30,"accuracy"),
        ("Глубина и полнота","Охватывает ли все аспекты задачи",0.20,"accuracy"),
        ("Логичность и структура","Структурированность и последовательность вывода",0.15,"accuracy"),
        ("Гибкость интерпретации","Работа с неоднозначными входными данными",0.15,"accuracy"),
        ("Устойчивость к шуму","Стабильность при зашумлённых входных данных",0.20,"robustness"),
        ("Обработка сложных задач","Работа с многоэтапными и составными запросами",0.15,"robustness"),
        ("Скорость ответа","Время инференса на CPU",0.10,"robustness"),
        ("Контекстная согласованность","Сохранение связи с контекстом задачи",0.15,"context"),
        ("Адаптивность","Подстройка под специфику конкретной задачи",0.10,"context"),
        ("Компактность модели","Размер модели и требования к памяти",0.10,"context"),
    ]
    for nm,ds,w,g in defaults:
        conn.execute("INSERT INTO criteria (project_id,name,description,weight,group_name) VALUES (?,?,?,?,?)",(pid,nm,ds,w,g))
    for m in models_cfg:
        conn.execute("INSERT INTO ai_models (project_id,name,model_type,description) VALUES (?,?,?,?)",
                     (pid,m['name'],m['model_type'],''))
    crit = {r['name']:r['id'] for r in conn.execute("SELECT id,name FROM criteria WHERE project_id=?", (pid,)).fetchall()}
    mods = {r['name']:r['id'] for r in conn.execute("SELECT id,name FROM ai_models WHERE project_id=?", (pid,)).fetchall()}
    for mn, sc in scores_data.items():
        mid = mods.get(mn)
        if not mid: continue
        for cn, sv in sc.items():
            cid = crit.get(cn)
            if cid:
                conn.execute("INSERT OR REPLACE INTO scores (model_id,criterion_id,score) VALUES (?,?,?)",
                             (mid,cid,sv))
    # Сразу считаем K чтобы при первом открытии всё было готово
    results = calculate_k(pid, conn)
    conn.execute("INSERT INTO results (project_id,result_json) VALUES (?,?)",
                 (pid, json.dumps(results)))
    conn.execute("UPDATE projects SET last_calculated=datetime('now'),report_status='ready' WHERE id=?",
                 (pid,))
    print(f"  ✓ Засеян проект: {name[:50]}")


def auto_seed_if_empty():
    """Если БД пустая — наполнить демо-проектами."""
    with get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        if count > 0:
            return
        print("[ Авто-наполнение демо-данными... ]")
        _seed_project(conn,
            'Выбор модели детекции для производства',
            'Сравнение YOLO-моделей для детекции дефектов. Приоритет: скорость + точность на CPU.',
            [{'name':'YOLOv8n','model_type':'detection'},{'name':'YOLOv8s','model_type':'detection'},
             {'name':'YOLOv8m','model_type':'detection'},{'name':'EfficientDet-D0','model_type':'detection'},
             {'name':'YOLOv5su','model_type':'detection'}],
            SCORES_DETECTION)
        _seed_project(conn,
            'NLP модели — анализ тональности отзывов',
            'Сравнение моделей сентимент-анализа для русскоязычных отзывов клиентов. '
            'Для локального тестирования: вкладка Тестирование → NLP-модели.',
            # Имена ДОЛЖНЫ совпадать с ключами NLP_MODEL_REGISTRY для автотестирования
            [{'name':'rubert-tiny2','model_type':'text','description':'29MB — быстрая, cointegrated/rubert-tiny2'},
             {'name':'rubert-base','model_type':'text','description':'180MB — точная, blanchefort/rubert-base-cased-sentiment'},
             {'name':'roberta-sentiment','model_type':'text','description':'125MB — cardiffnlp/twitter-roberta'},
             {'name':'distilbert-ru','model_type':'text','description':'68MB — multilingual distilbert'}],
            SCORES_NLP)
        print("[ Демо-данные готовы. Открой http://localhost:8000 ]")

auto_seed_if_empty()


# ════════════════════════════════════════════════════════════════════════════
# ОТДАЧА ФРОНТЕНДА
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Отдаёт index.html из ../frontend/. Это единственная страница SPA."""
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found</h1>")


# ════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn, threading, webbrowser, time

    def open_browser():
        """Открыть браузер автоматически через 1.5 сек после старта."""
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
