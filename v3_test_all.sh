#!/usr/bin/env bash

echo "v3 test - no finetuning"
sh test_v3_resnet.sh
echo "v3 test - retraining (finetuning)"
sh retrain_v3_resnet.sh
echo "v3 test - training"
sh run_v3_resnet.sh