#!/usr/bin/env bash

echo "v2q test - no finetuning"
sh v2q.sh
echo "v2q test - retraining (finetuning)"
sh v2q_retrain.sh
echo "v2q test - training"
sh v2q_run.sh
