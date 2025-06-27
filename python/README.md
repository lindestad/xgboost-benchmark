# Python implementation

## Benchmark results:

**Home Computer**
Ryzen 5900X (12 cores / 12 HT):
101.5s/iteration. 89 iterations => 152.8 minutes.

RTX 3070:
17.8s/iteration. 89 iterations => 26.4 minutes.

GPU speedup is 5.79x.

**Shared Compute Server**

Xeon Platinum 8280:

12 cores: 54s/it => 77.1 minutes.
12 cores with HT: 56s/it => 83 minutes.

28 cores: 48s/it => 71.2 minutes.
28 cores with HT: 53s/it => 78.6 minutes.

Split (multiple processes running the same workload, simulating hyperparam search):
28 cores - 2 jobs: 72.3s/it - 
combined 36.2s/it => 53.7 minutes.

28 cores - 4 jobs: 123s/it - 
combined 30.8s/it => 45.7 minutes.

28 cores - 8 jobs: 227.4s/it -
combined 28.4s/it => 42.1 minutes.

56 cores - 4 jobs (2 per NUMA node): 71.5s/it -
average per NUMA node: 35.75s/it
combined 17.88s/it => 26.5 minutes.

**Conclusion:**

A single NUMA node (28 cores) is about half as powerful (in this workload) as a single RTX 3070.

Two NUMA nodes (58 cores) is roughly equal to a single RTX 3070.

A full 112 core 8280 is similar in performance (for Hyperparam search) as two 3070s.

## Installing

Set up Python (3.12 used for benchmarks):

```bash
uv venv --python 3.12
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
uv pip sync requirements.lock
```

## Running the benchmark

Run the benchmark with:

```bash
python src/walkforward_cpu.py
python src/walkforward_gpu.py
```
