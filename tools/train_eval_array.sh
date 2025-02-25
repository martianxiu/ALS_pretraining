#!/bin/bash

#$-l rt_F=1
#$-l h_rt=24:00:00
#$-j y
#$-cwd
#$-t 1-3:1
#$-N bevmae_ft_opengf_vsize0.6_D2362W3_pureforest
#$-o /home/aac12888ea/OpenPCDet/ABCI_log/$JOB_NAME/$JOB_NAME_$TASK_ID.txt
#$-e /home/aac12888ea/OpenPCDet/ABCI_log/$JOB_NAME/$JOB_NAME_$TASK_ID.txt
#$-m abe 

source /etc/profile.d/modules.sh
module load cuda/12.1/12.1.1

source /home/aac12888ea/.bashrc
conda activate openpcdet_v0.6.0
  
nvidia-smi

## BEV-MAE train 
set -x
EXP_NAME=${JOB_NAME}_${TASK_ID}
string=${JOB_NAME}
CONFIG_NAME=${string##*_}_models/${JOB_NAME}.yaml
NGPUS=4
bash ./scripts/dist_train_val.sh $NGPUS \
    --cfg_file cfgs/$CONFIG_NAME --extra_tag $EXP_NAME \
    --ckpt_save_interval 1 \
    --workers 4 \
    --logger_iter_interval 5 \
    # --use_amp 

## BEV-MAE eval
string=${JOB_NAME}
CONFIG_FILE=../output/${string##*_}_models/${JOB_NAME}/${JOB_NAME}_${TASK_ID}/${JOB_NAME}.yaml
CKPT=../output/${string##*_}_models/${JOB_NAME}/${JOB_NAME}_${TASK_ID}/ckpt/checkpoint_best.pth

python test_seg.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CKPT} --workers 6
