#!/bin/sh
CUDA_VISIBLE_DEVICES=6 python -m scripts.train --env MiniGrid-DoorKey-5x5-v0 --model DoorKeypposhapcoeff0.7 --algo pposhap --frames 100000 --shap_coeff 0.7