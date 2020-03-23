#!/usr/bin/env bash

echo "v3 test - no finetuning"
sh v3_resnet.sh
echo "v3 test - retraining (finetuning)"
sh v3_resnet_retrain.sh
echo "v3 test - training"
sh v3_resnet_run.sh
