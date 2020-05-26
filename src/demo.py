from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing, re_mkdir
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import torch


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    # mkdir_if_missing(result_root)
    re_mkdir(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    try:
        eval_seq(opt, dataloader, 'mot', result_filename,
                 save_dir=frame_dir, show_image=False, frame_rate=frame_rate)
    except Exception as e:
        logger.info(e)

    if opt.output_format == 'video':
        # output_video_path = osp.join(result_root, 'result.mp4')
        output_video_path = osp.join('/workspace/FairMOT/result-videos', opt.input_video.split('/')[-1])
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opt = opts().init()
    print('opt.gpus_str', opt.gpus_str)
    print('opt.gpus', opt.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print('opt.device', opt.device)
    demo(opt)
