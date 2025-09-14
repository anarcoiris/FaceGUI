# core/db.py
import sqlite3
import os
import json
import time

DB_FILE = os.environ.get('DB_FILE', './face_gui.sqlite')

def init_db(db_file=DB_FILE):
    os.makedirs(os.path.dirname(db_file) or '.', exist_ok=True)
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS configs (
        name TEXT PRIMARY KEY,
        cfg_json TEXT,
        created_at TEXT
    )''')
    cur.execute('''CREATE TABLE IF NOT EXISTS mappings (
        personId TEXT,
        persistedFaceId TEXT,
        blob_url TEXT,
        uploaded_at TEXT
    )''')
    conn.commit()
    conn.close()

def save_config(name, cfg, db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('REPLACE INTO configs (name, cfg_json, created_at) VALUES (?, ?, ?)', (name, json.dumps(cfg), time.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def list_configs(db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('SELECT name, created_at FROM configs ORDER BY created_at DESC')
    rows = cur.fetchall()
    conn.close()
    return rows

def save_mapping(personId, persistedFaceId, blob_url, db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('INSERT INTO mappings (personId, persistedFaceId, blob_url, uploaded_at) VALUES (?,?,?,?)', (personId, persistedFaceId, blob_url, time.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()
