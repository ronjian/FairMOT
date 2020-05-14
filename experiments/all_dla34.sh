cd src
python train.py \
mot \
--arch dla_34 \
--exp_id all_dla34_7 \
--head_conv 256 \
--gpus 0,1 \
--batch_size 8 \
--load_model '../models/ctdet_coco_dla_2x.pth'
cd ..