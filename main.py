from fastapi import FastAPI, Query
from searcher import Searcher
from indexer import Indexer
import sqlite3
import json

app = FastAPI()
conn = sqlite3.connect("imgs.sqlite")
searcher = Searcher(conn)
indexer = Indexer(searcher)

@app.get("/search")
def search(fileId: str = Query(...), limit: int = Query(5), offset: int = Query(0), weights: str = Qeury("{}")):
    return searcher.search(fileId, limit, offset, json.loads(weights))

@app.post("/update")
def update(data: dict):
    conn = sqlite3.connect("imgs.sqlite")
    result = indexer.add(conn, data["fileId"], data["url"], data["searchable"], data["duplicate"])
    conn.close()
    return result
