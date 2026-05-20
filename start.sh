#!/bin/bash
# ── AI CharacherHub — Quick Start ───────────────────────────────────────────

echo "=== AI CharacherHub ==="
echo ""

# 1. Install deps
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q

# 2. Seed demo data (optional)
if [ "$1" == "--demo" ]; then
  echo "[2/3] Seeding demo project..."
  cd backend && python seed_demo.py && cd ..
else
  echo "[2/3] Skipping demo data (run with --demo to seed)"
fi

# 3. Start server
echo "[3/3] Starting server at http://localhost:8000"
echo ""
echo "  Open: http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo ""
cd backend && python main.py
