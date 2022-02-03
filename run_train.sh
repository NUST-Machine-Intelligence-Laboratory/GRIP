#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONWARNINGS="ignore"

# resnet18, resnet50Health Screening Form - MC1
export NET='resnet50'
export path='model'
export data='/home/zcy/data/fg-web-data/web-bird'
export N_CLASSES=200
export lr=0.01
export w_decay=1e-5
export epochs=100
export batchsize=32

python train.py --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data} --lr ${lr} --w_decay ${w_decay} --batch_size ${batchsize} --epochs ${epochs} --nh 0.5 --warmup 5 --denoise --alpha 0.5 --relabel --tau 0.04

sleep 100
python train.py --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data} --lr ${lr} --w_decay ${w_decay} --batch_size ${batchsize} --epochs ${epochs} --nh 0.5 --warmup 5 --denoise --alpha 0.5 --relabel --tau 0.04

sleep 100
python train.py --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data} --lr ${lr} --w_decay ${w_decay} --batch_size ${batchsize} --epochs ${epochs} --nh 0.5 --warmup 5 --denoise --alpha 0.5 --relabel --tau 0.04


