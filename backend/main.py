"""
AI CharacherHub — Backend
FastAPI + SQLite, no external DB required
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator, Field
from typing import Optional, List
from pathlib import Path
import sqlite3, json, csv, io, os
from datetime import datetime

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI CharacherHub", version="1.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB_PATH = os.getenv("DB_PATH", "ai_eval.db")

# ── DB ───────────────────────────────────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                description     TEXT    DEFAULT '',
                created_at      TEXT    DEFAULT (datetime('now')),
                last_calculated TEXT,
                report_status   TEXT    DEFAULT 'pending'
            );
            CREATE TABLE IF NOT EXISTS ai_models (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name        TEXT    NOT NULL,
                model_type  TEXT    NOT NULL DEFAULT 'custom',
                description TEXT    DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS criteria (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name        TEXT    NOT NULL,
                description TEXT    DEFAULT '',
                weight      REAL    NOT NULL DEFAULT 0.1,
                group_name  TEXT    NOT NULL DEFAULT 'accuracy',
                enabled     INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS scores (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id     INTEGER NOT NULL REFERENCES ai_models(id)  ON DELETE CASCADE,
                criterion_id INTEGER NOT NULL REFERENCES criteria(id)   ON DELETE CASCADE,
                score        REAL    NOT NULL CHECK(score >= 1 AND score <= 5),
                UNIQUE(model_id, criterion_id)
            );
            CREATE TABLE IF NOT EXISTS results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                calculated_at TEXT    DEFAULT (datetime('now')),
                result_json   TEXT    NOT NULL
            );
        """)

init_db()

# ── Schemas ──────────────────────────────────────────────────────────────────
SAFE_CHARS = set('<>;\'"\\/')

class ProjectCreate(BaseModel):
    name: str
    description: str = ""

class ModelCreate(BaseModel):
    name: str
    model_type: str = "custom"
    description: str = ""

    @validator('name')
    def safe_name(cls, v):
        if any(c in v for c in SAFE_CHARS):
            raise ValueError('Invalid characters in name')
        return v.strip()

class CriterionCreate(BaseModel):
    name: str
    description: str = ""
    weight: float = Field(0.1, ge=0.0, le=1.0)
    group_name: str = "accuracy"

    @validator('name')
    def safe_name(cls, v):
        if any(c in v for c in SAFE_CHARS):
            raise ValueError('Invalid characters in name')
        return v.strip()

class CriterionUpdate(BaseModel):
    name:        Optional[str]   = None
    description: Optional[str]   = None
    weight:      Optional[float] = Field(None, ge=0.0, le=1.0)
    group_name:  Optional[str]   = None
    enabled:     Optional[int]   = None

class ScoreSet(BaseModel):
    model_id:     int
    criterion_id: int
    score:        float = Field(..., ge=1.0, le=5.0)

class SensitivityRequest(BaseModel):
    criterion_id: int
    delta:        float = Field(0.1, ge=0.01, le=0.5)

# ── Math engine ──────────────────────────────────────────────────────────────
def interpret_k(k: float) -> str:
    if k >= 0.90: return "Отличная модель — рекомендуется"
    if k >= 0.75: return "Хорошая модель"
    if k >= 0.60: return "Приемлемая модель"
    if k >= 0.40: return "Слабая модель"
    return "Не рекомендуется"

def calculate_k(project_id: int, conn) -> dict:
    models   = conn.execute("SELECT * FROM ai_models WHERE project_id=?", (project_id,)).fetchall()
    criteria = conn.execute(
        "SELECT * FROM criteria WHERE project_id=? AND enabled=1", (project_id,)
    ).fetchall()

    if not models or not criteria:
        return {}

    results = {}
    for model in models:
        s_k = 0.0
        s_max = 0.0
        group_details = {}

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

            group_details[g] = {
                "k":      round(g_s_k / g_s_max, 4) if g_s_max > 0 else 0.0,
                "s_k":    round(g_s_k, 4),
                "s_max":  round(g_s_max, 4),
                "criteria": crit_rows,
            }

        k = round(s_k / s_max, 4) if s_max > 0 else 0.0

        # fill contribution %
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

    # Assign ranks
    sorted_ids = sorted(results.keys(), key=lambda mid: -results[mid]['k'])
    for rank, mid in enumerate(sorted_ids, 1):
        results[mid]['rank'] = rank

    return results

def run_sensitivity(project_id: int, criterion_id: int, delta: float, conn) -> dict:
    baseline = calculate_k(project_id, conn)
    if not baseline:
        return {}

    bl_leader = min(baseline.keys(), key=lambda m: baseline[m]['rank'])
    orig = conn.execute("SELECT weight FROM criteria WHERE id=?", (criterion_id,)).fetchone()
    if not orig:
        return {}

    orig_w  = orig['weight']
    new_w   = min(1.0, orig_w + delta)
    conn.execute("UPDATE criteria SET weight=? WHERE id=?", (new_w, criterion_id))
    modified = calculate_k(project_id, conn)
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

# ── Security helpers ──────────────────────────────────────────────────────────
def validate_project_exists(pid: int, conn):
    if not conn.execute("SELECT 1 FROM projects WHERE id=?", (pid,)).fetchone():
        raise HTTPException(404, f"Project {pid} not found")

# ────────────────────────────────────────────────────────────────────────────
# ROUTES — Projects
# ────────────────────────────────────────────────────────────────────────────
@app.get("/api/projects")
def list_projects():
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
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO projects (name, description) VALUES (?,?)",
            (data.name.strip(), data.description)
        )
        pid = cur.lastrowid
        # Default criteria
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
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        p = dict(conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone())
        p['models']   = [dict(m) for m in conn.execute("SELECT * FROM ai_models WHERE project_id=?", (pid,)).fetchall()]
        p['criteria'] = [dict(c) for c in conn.execute("SELECT * FROM criteria WHERE project_id=? ORDER BY group_name,id", (pid,)).fetchall()]
        return p

@app.delete("/api/projects/{pid}")
def delete_project(pid: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM projects WHERE id=?", (pid,))
    return {"ok": True}

# ── AI Models ────────────────────────────────────────────────────────────────
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

# ── Criteria ─────────────────────────────────────────────────────────────────
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
    updates = {k: v for k, v in data.dict().items() if v is not None}
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
    """Normalize weights so they sum to 1.0 per group"""
    with get_conn() as conn:
        crit = conn.execute("SELECT * FROM criteria WHERE project_id=? AND enabled=1", (pid,)).fetchall()
        groups = {}
        for c in crit:
            groups.setdefault(c['group_name'], []).append(c)
        for g_list in groups.values():
            total = sum(c['weight'] for c in g_list)
            if total > 0:
                for c in g_list:
                    conn.execute("UPDATE criteria SET weight=? WHERE id=?", (round(c['weight']/total, 4), c['id']))
    return {"ok": True}

# ── Scores ────────────────────────────────────────────────────────────────────
@app.get("/api/projects/{pid}/scores")
def get_scores(pid: int):
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
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO scores (model_id,criterion_id,score) VALUES (?,?,?)
            ON CONFLICT(model_id,criterion_id) DO UPDATE SET score=excluded.score
        """, (data.model_id, data.criterion_id, data.score))
    return {"ok": True}

@app.post("/api/projects/{pid}/scores/import")
async def import_csv(pid: int, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, "Only CSV files allowed")
    content = await file.read()
    if len(content) > 2 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 2MB)")

    text   = content.decode('utf-8-sig')
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
                    errors.append(f"Row {i}: score {score} out of range [1,5]"); continue
                model = conn.execute(
                    "SELECT id FROM ai_models WHERE project_id=? AND name=?",
                    (pid, row['model_name'].strip())
                ).fetchone()
                crit = conn.execute(
                    "SELECT id FROM criteria WHERE project_id=? AND name=?",
                    (pid, row['criterion_name'].strip())
                ).fetchone()
                if not model: errors.append(f"Row {i}: model '{row['model_name']}' not found"); continue
                if not crit:  errors.append(f"Row {i}: criterion '{row['criterion_name']}' not found"); continue
                conn.execute("""
                    INSERT INTO scores (model_id,criterion_id,score) VALUES (?,?,?)
                    ON CONFLICT(model_id,criterion_id) DO UPDATE SET score=excluded.score
                """, (model['id'], crit['id'], score))
                imported += 1
            except (ValueError, KeyError) as e:
                errors.append(f"Row {i}: {e}")
    return {"imported": imported, "errors": errors}

# ── Calculation ───────────────────────────────────────────────────────────────
@app.post("/api/projects/{pid}/calculate")
def run_calculation(pid: int):
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        results = calculate_k(pid, conn)
        if not results:
            raise HTTPException(400, "No models or criteria. Add them first.")
        conn.execute(
            "INSERT INTO results (project_id,result_json) VALUES (?,?)",
            (pid, json.dumps(results))
        )
        conn.execute(
            "UPDATE projects SET last_calculated=datetime('now'), report_status='ready' WHERE id=?",
            (pid,)
        )
        return results

@app.get("/api/projects/{pid}/results")
def get_results(pid: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM results WHERE project_id=? ORDER BY id DESC LIMIT 1", (pid,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "No results yet. Run calculation first.")
        return {"calculated_at": row['calculated_at'], "results": json.loads(row['result_json'])}

# ── Sensitivity ───────────────────────────────────────────────────────────────
@app.post("/api/projects/{pid}/sensitivity")
def sensitivity(pid: int, data: SensitivityRequest):
    with get_conn() as conn:
        result = run_sensitivity(pid, data.criterion_id, data.delta, conn)
        if not result:
            raise HTTPException(400, "Could not run analysis. Ensure scores are entered.")
        return result

# ── Report ────────────────────────────────────────────────────────────────────
@app.get("/api/projects/{pid}/report")
def get_report(pid: int):
    with get_conn() as conn:
        validate_project_exists(pid, conn)
        latest = conn.execute(
            "SELECT * FROM results WHERE project_id=? ORDER BY id DESC LIMIT 1", (pid,)
        ).fetchone()
        if not latest:
            raise HTTPException(404, "No results. Run calculation first.")

        proj    = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        results = json.loads(latest['result_json'])
        sorted_models = sorted(results.values(), key=lambda x: x['rank'])
        winner  = sorted_models[0]

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

# ── Serve frontend ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found</h1><p>Put index.html in /frontend/</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
