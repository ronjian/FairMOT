cd src
nohup python -u train.py \
mot \
--arch mobilenetv2_7 \
--exp_id all_mobilenetv2_7 \
--head_conv 256 \
--gpus 0,1 \
--batch_size 8 > all_mobilenetv2_7_training.log 2>&1 &
cd ..