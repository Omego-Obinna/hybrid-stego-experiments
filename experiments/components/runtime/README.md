# Current QIM runtime and overhead

This package uses the existing 10,000-image runtime summary, optional per-image logs, and existing stego files. It never calls the embedder or decoder.

Run:

```bash
cd /home/omego/Documents/Hybrid_Stego_Eva
source /home/omego/Documents/Hybrid_Stego_Eva/myenv/bin/activate
bash /home/omego/Documents/Hybrid_Stego_Eva/current_qim_runtime_overhead/run_current_qim_runtime_overhead.sh
```

The summary CSV is sufficient. Raw logs are optional and add median/P95/min/max statistics.
