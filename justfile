# generate config
config:
    pkl eval -f json -o configs/main.json configs/main.pkl

# warmup
warmup:
    python run.py warmup

# warmup-checkpoint
warmup-checkpoint:
    python run.py warmup-checkpoint

# test
test:
    python run.py test

# test-checkpoint
test-checkpoint:
    python run.py test-checkpoint

# eval
eval:
    python run.py eval

# format
format:
    ruff format .
    ruff check --fix --select I

# format test
format-test:
    ruff format test
