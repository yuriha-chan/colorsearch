import pickle
import csv
import heapq
import json

import numpy as np

def top_n_by_score(a, f, n):
    heap = []
    for x in a:
        score = f(x)
        print(x["fileId"], " ", score)
        if len(heap) < n:
            heapq.heappush(heap, (score, x))
        else:
            # compare with samllest value in the heap
            if score > heap[0][0]:
                # push the new pair and pop the smallest pair
                heapq.heappushpop(heap, (score, x))
    print([s for s, _ in heap])
    return [x for _, x in heap]

def features(e):
    v = np.log(1+np.array(e["accentColors"]["features"]))
    return v / (1e-12 + np.linalg.norm(v))

def dist(a, b):
    return 1 - np.dot(a, b)

class Searcher:
    def __init__(self, db):
        res = db.cursor().execute("SELECT * from image")
        v = res.fetchall()
        self.data = {e[0]: json.loads(e[1]) for e in v}
        self.data_searchable = [json.loads(e[1]) for e in v if e[2]]

    def has(self, fileId):
        return fileId in self.data

    def search(self, fileId, n, m = 0, weights = {}):
        x = self.data[fileId]
        fx = features(x)
        score_func = lambda y: -(weights.get("features", 1) * dist(fx, features(y)) + weights.get("entropy", 0.5) * abs(x["entropy"] - y["entropy"]) + weights.get("maxChroma", 0.2) * abs(x["maxChroma"] - y["maxChroma"]) + weights.get("meanLuminocity", 0.2) * abs(x["meanLuminocity"] - y["meanLuminocity"]))
        top_n = top_n_by_score(self.data_searchable, score_func, n + m)
        return [{"fileId": r["fileId"]} for r in top_n][m:]