#!/usr/bin/env bash

echo "v2a test - no finetuning"
sh test_v2a_resnet.sh
echo "v2a test - retraining (finetuning)"
sh retrain_v2a_resnet.sh
echo "v2a test - training"
sh run_v2a_resnet.sh