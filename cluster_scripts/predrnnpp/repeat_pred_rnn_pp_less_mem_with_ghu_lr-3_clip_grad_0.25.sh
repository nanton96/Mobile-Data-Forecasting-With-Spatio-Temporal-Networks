/#!/bin/sh

for i in {0..88..8}
    do
    sbatch cluster_scripts/predrnnpp/pred_rnn_pp_less_mem_with_ghu_lr-3_clip_grad_0.25.sh $i
    done






