#!/usr/bin/env bash

echo "v3a test - no finetuning"
sh test_v3a_resnet.sh
echo "v3a test - retraining (finetuning)"
sh retrain_v3a_resnet.sh
echo "v3a test - training"
sh run_v3a_resnet.sh