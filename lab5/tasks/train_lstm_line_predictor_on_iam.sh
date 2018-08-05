#!/bin/sh
pipenv run python training/run_experiment.py --save '{"dataset": "IamLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc", "network_args": {"bidirectional": 1, "lstm_dim": 64, "window_stride": 2, "window_width": 14}, "train_args": {"batch_size": 32, "epochs": 40}}'
