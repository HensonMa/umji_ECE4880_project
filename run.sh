#!/usr/bin/env bash

python train_normal.py --model-name=vgg19_quantized --type=imagenet --num-class=200 --bit-width=8 --depth 1 2 3 4 5
python train_normal.py --model-name=vgg19_quantized --type=imagenet --num-class=200 --bit-width=6 --depth 1 2 3 4 5
python train_normal.py --model-name=vgg19_quantized --type=imagenet --num-class=200 --bit-width=4 --depth 1 2 3 4 5
python train_normal.py --model-name=vgg19_quantized --type=imagenet --num-class=200 --bit-width=2 --depth 1 2 3 4 5
python train_normal.py --model-name=vgg19_quantized --type=imagenet --num-class=200 --bit-width=4 --depth 3 4 5
python train_normal.py --model-name=vgg19_quantized --type=imagenet --num-class=200 --bit-width=4 --depth 4 5
python train_normal.py --model-name=vgg19_quantized --type=imagenet --num-class=200 --bit-width=4 --depth=5
