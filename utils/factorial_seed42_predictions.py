"""Re-run the seed-42 factorial runs (script 19, deterministic GPU) to store their test-set predictions for the
illustrative confusion-matrix and ROC figures; asserts that every re-run reproduces the stored metrics exactly.
Output: results/revision_factorial_seed42_predictions.json"""
import importlib.util, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('s19', os.path.join(ROOT, '19_revision_experiments.py'))
s19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s19)
stored = {(r['machine'], r['cfg']['rep'], r['cfg']['weighted']): r for exp in ('equiv', 'factorial')
          for r in json.load(open(os.path.join(ROOT, 'results', f'revision_{exp}.json'))) if r['seed'] == 42}
out = []
for m in s19.MACHINES:
    for rep in ('mfcc', 'logmel'):
        X, y, _ = s19.load(m, 'm6', rep)
        for w in (False, True):
            r = s19.standard_run(X, y, m, 42, '3B-FC128', w, X.shape[2])
            ref = stored[(m, rep, w)]
            assert all(r[k] == ref[k] for k in ('auc', 'pauc', 'f1', 'f1_cal', 'tau')), (m, rep, w, r['auc'], ref['auc'])
            out.append(dict(machine=m, rep=rep, weighted=w, seed=42, auc=r['auc'], y_true=r['y_true'], y_prob=r['y_prob']))
            print(m, rep, 'W' if w else 'B', f"AUC {r['auc']:.4f} (identical to stored run)", flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', 'revision_factorial_seed42_predictions.json'), 'w'))
print('saved')
