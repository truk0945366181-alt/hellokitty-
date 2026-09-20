"""
event_log.py
============
บันทึกและอ่านประวัติเหตุการณ์ (Event Log) เก็บเป็นไฟล์ CSV ง่าย ๆ
คอลัมน์: timestamp, status (known/unknown), name, confidence, image_path, alert_sent
"""

import os
import pandas as pd
from datetime import datetime

LOG_CSV = "data/logs/event_log.csv"
COLUMNS = ["timestamp", "status", "name", "confidence", "image_path", "alert_sent"]


def _ensure_log_file():
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    if not os.path.exists(LOG_CSV):
        pd.DataFrame(columns=COLUMNS).to_csv(LOG_CSV, index=False)


def add_event(status, name, confidence, image_path, alert_sent):
    _ensure_log_file()
    df = pd.read_csv(LOG_CSV)
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "name": name if name else "-",
        "confidence": round(confidence, 1) if confidence is not None else "-",
        "image_path": image_path,
        "alert_sent": alert_sent,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_CSV, index=False)


def get_all_events():
    _ensure_log_file()
    df = pd.read_csv(LOG_CSV)
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


def get_summary():
    df = get_all_events()
    total = len(df)
    unknown = len(df[df["status"] == "unknown"])
    known = len(df[df["status"] == "known"])
    return {"total": total, "unknown": unknown, "known": known}
