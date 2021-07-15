import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import os
import sys
import time
import datetime
import shutil
import copy
import random
import math
import json
import pickle
import logging
import logging.handlers
import torch.utils.data.distributed
import torch.nn.functional as F
import yaml
from common.util import Options
from models.blocks import *
from models.darknight import DarkNight
with open("config.yaml", "r") as ymlfile:
    options = Options(yaml.load(ymlfile, Loader=yaml.FullLoader)['options'])
model = DarkNight(options)