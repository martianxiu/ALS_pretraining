## BEV-MAE train 
JOB_NAME=bevmae_pretrain_mask0.7_D2362W3_sample40_als
TASK_ID=100

set -x
EXP_NAME=${JOB_NAME}_${TASK_ID}
string=${JOB_NAME}
CONFIG_NAME=${string##*_}_models/${JOB_NAME}.yaml
NGPUS=1
bash ./scripts/dist_train_val.sh $NGPUS \
    --cfg_file cfgs/$CONFIG_NAME --extra_tag $EXP_NAME \
    --ckpt_save_interval 1 \
    --workers 4 \
    --logger_iter_interval 5\
    --val_interval 0