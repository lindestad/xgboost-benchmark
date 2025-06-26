# Python implementation

Benchmark results:

Ryzen 5900X:
101.5s/iteration. 89 iterations => 152.8 minutes.

RTX 3070:
17.8s/iteration. 89 iterations => 26.4 minutes.

GPU speedup is 5.79x.

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
