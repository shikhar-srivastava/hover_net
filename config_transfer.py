import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self, source_organ, target_organ, bucket_step):
        self.seed = 10

        # Adrenal_gland  Bile-duct  Breast  Cervix  Colon  Esophagus  HeadNeck  Liver  Thyroid
        self.source_organ = source_organ 
        self.target_organ = target_organ
        self.bucket_step = bucket_step
        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "hovernet"
        model_mode = "fast" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = 6 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [164, 164] # patch shape at output of network

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [164,164]:
                raise Exception("If using `original` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "pannuke" # extracts dataset info from dataset.py
        self.log_dir = "/l/users/shikhar.srivastava/workspace/hover_net/logs/second_order/%s/ckpts/%s-%s/" % (self.bucket_step, self.source_organ, self.target_organ) # where checkpoints will be saved

        # For Transfer, training and validation lists are both the validation sets for the target organ

        self.train_dir_list = [
            "/l/users/shikhar.srivastava/data/pannuke/processed/%s/%s/test/540x540_164x164/" % (self.bucket_step, self.target_organ)
        ]
        self.valid_dir_list = [
            "/l/users/shikhar.srivastava/data/pannuke/processed/%s/%s/test/540x540_164x164/" % (self.bucket_step, self.target_organ)
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt_second_order" % model_name
        )

        pretrained_path = f'/l/users/shikhar.srivastava/workspace/hover_net/logs/first_order/{self.bucket_step}/ckpts/{self.source_organ}/01/net_epoch=50.tar'
        self.model_config = module.get_config(nr_type, model_mode, pretrained_path)
