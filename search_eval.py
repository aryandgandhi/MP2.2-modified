import math
import sys
import time
import metapy
import pytoml


def compute_ndcg_for_params(idx, ranker, query_cfg, cfg_file, top_k):
    ev = metapy.index.IREval(cfg_file)
    ndcg = 0.0
    num_queries = 0
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)
    query = metapy.index.Document()

    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            ndcg += ev.ndcg(results, query_start + query_num, top_k)
            num_queries += 1

    return ndcg / num_queries

def tune_parameters(idx, query_cfg, cfg_file):
    best_ndcg = -1
    best_params = (1.2, 0.75)
    for k1 in [1.0, 1.2, 1.5, 1.8, 2.0]:
        for b in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            ranker = metapy.index.OkapiBM25(k1=k1, b=b, k3=500)
            ndcg = compute_ndcg_for_params(idx, ranker, query_cfg, cfg_file, top_k=10)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_params = (k1, b)
    return best_params

def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    """
    idx = metapy.index.make_inverted_index(cfg_file)
    with open(cfg_file, 'r') as fin:
        cfg_d = pytoml.load(fin)
    query_cfg = cfg_d['query-runner']

    best_k1, best_b = tune_parameters(idx, query_cfg)
    return metapy.index.OkapiBM25(k1=best_k1, b=best_b, k3=500)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    query = metapy.index.Document()
    ndcg = 0.0
    num_queries = 0

    print('Running queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            ndcg += ev.ndcg(results, query_start + query_num, top_k)
            num_queries+=1
    ndcg= ndcg / num_queries
            
    print("NDCG@{}: {}".format(top_k, ndcg))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
