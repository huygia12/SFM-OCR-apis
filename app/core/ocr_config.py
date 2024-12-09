import torch
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def get_vietocr_predictor():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['predictor']['beamsearch'] = False
    config['cnn']['pretrained'] = False
    return Predictor(config)