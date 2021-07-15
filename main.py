import yaml
from .common.util import Options
from .models.blocks import *
from .models.darknight import DarkNight
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, type=str, help='train or ctrain or test')
    with open("config.yaml", "r") as ymlfile:
        options = Options(yaml.load(ymlfile, Loader=yaml.FullLoader)['options'])
    model = DarkNight(options)
        