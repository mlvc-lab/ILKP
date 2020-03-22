#!/usr/bin/env bash

echo "v2q 8bit test - no finetuning"
sh test_v2q_resnet.sh
echo "v2q 8bit test - retraining (finetuning)"
sh retrain_v2q_resnet.sh
echo "v2q 8bit test - training"
sh run_v2q_resnet.sh