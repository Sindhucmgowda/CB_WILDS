#!/bin/bash 

srun -p interactive -q nopreemption --mem=32G --gres=gpu:1 --pty bash