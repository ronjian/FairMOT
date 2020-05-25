cd src
nohup python -u train.py \
mot \
--arch mobilenetv2_7 \
--exp_id all_mobilenetv2_7 \
--head_conv 256 \
--num_epochs 70 \
--lr_step 40,54 \
--gpus 0,1 \
--reid_dim 128 \
--batch_size 8 > all_mobilenetv2_7_training.log 2>&1 &
cd ..


python demo.py \
mot \
--exp_id all_mobilenetv2_7 \
--head_conv 256 \
--arch mobilenetv2_7 \
--reid_dim 128 \
--gpu 1 \
--input-video ../videos/MOT17-03.mp4 \
--load_model /workspace/FairMOT/exp/mot/all_mobilenetv2_7/model_last.pth \
--conf_thres 0.4
