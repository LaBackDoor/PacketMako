#!/bin/bash
source traicenv/bin/activate
module load python39
# to train
# python main.py train model/.config /data/arguellesa/traice/output
# to test
python main.py model/.config /data/arguellesa/traice/old_weights/checkpoint-500 /data/arguellesa/traice/1.txt
