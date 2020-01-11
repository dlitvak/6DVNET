# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("--config-file", default="../configs/e2e_3d_car_101_FPN_triple_head.yaml", metavar="FILE", help="path to config file", type=str)
    # parser.add_argument("--weight", default="/media/SSD_1TB/ApolloScape/6DVNET_experiments/e2e_3d_car_101_FPN_triple_head/May27-05-43_n606_step/model_final.pth")
    parser.add_argument("--weight", default="../models/e2e_mask_rcnn_R_101_FPN_1x.pth")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    args = parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(["MODEL.WEIGHT", args.weight])

    output_dir = os.path.dirname(cfg.MODEL.WEIGHT)
    cfg.OUTPUT_DIR = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.MODEL.WEIGHT)
    _ = checkpointer.load(cfg.MODEL.WEIGHT, cfg.TRAIN.IGNORE_LIST)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    # default `log_dir` is "runs" - we'll be more specific here
    # tb_writer = SummaryWriter('runs/6dvnet_test_3d_1')

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        # dataiter = iter(data_loader_val)
        # images, bbox, labels = dataiter.next()

        # create grid of images
        # img_grid = make_grid(images.tensors)

        # show images
        # matplotlib_imshow(img_grid, one_channel=False)

        # write to tensorboard
        # tb_writer.add_image('6dvnet_test_3d_1', img_grid)
        #
        # tb_writer.add_graph(model, images.tensors)
        # tb_writer.close()

        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            cfg=cfg,
        )
        synchronize()

if __name__ == "__main__":
    main()
