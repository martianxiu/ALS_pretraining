## BEV-MAE train 
JOB_NAME=bevmae_vsize0.6_D2362W3_pureforest
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
    --logger_iter_interval 5

## BEV-MAE eval
string=${JOB_NAME}
CONFIG_FILE=../output/${string##*_}_models/${JOB_NAME}/${JOB_NAME}_${TASK_ID}/${JOB_NAME}.yaml
CKPT=../output/${string##*_}_models/${JOB_NAME}/${JOB_NAME}_${TASK_ID}/ckpt/checkpoint_best.pth
python test_seg.py --cfg_file ${CONFIG_FILE} --batch_size 1 --ckpt ${CKPT} --workers 6
