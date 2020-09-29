# -*-coding:utf-8-*-

from .alexnet import *


def get_model(config):
    return globals()[config.arch](config.num_classes)
