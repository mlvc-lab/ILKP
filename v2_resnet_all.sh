#!/usr/bin/env bash

echo "v2 test - no finetuning"
sh v2_resnet.sh
echo "v2 test - retraining (finetuning)"
sh v2_resnet_retrain.sh
echo "v2 test - training"
sh v2_resnet_run.sh
