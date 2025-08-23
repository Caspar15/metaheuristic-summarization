#!/usr/bin/env bash
set -e

python -m src.data.preprocess --config configs/dataset_cnn_dm.yaml
python -m src.pipeline.build_features --config configs/features_basic.yaml
python -m src.pipeline.select_sentences --config configs/features_basic.yaml --method greedy
python -m src.pipeline.evaluate_run --config configs/eval.yaml

