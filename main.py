import yaml
from common.util import Options
from models.blocks import *
from models.darknight import DarkNight
with open("config.yaml", "r") as ymlfile:
    options = Options(yaml.load(ymlfile, Loader=yaml.FullLoader)['options'])
model = DarkNight(options)