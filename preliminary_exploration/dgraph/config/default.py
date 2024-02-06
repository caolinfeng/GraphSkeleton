from yacs.config import CfgNode as CN

_C = CN()
_C.Name = ''
_C.hidden_channels = 256
_C.num_layers = 2
_C.epochs = 200
_C.lr = 0.001
_C.dropout = 0.0
_C.weight_decay = 0.0
_C.log_steps = 1
_C.batchnorm = False
_C.batch_size = 65536
_C.sample_size = [-1,-1,-1]

def update_config(cfg, model):
    cfg.defrost()
    cfg_file = './config/'+model+'.yaml'
    cfg.merge_from_file(cfg_file)
    cfg.freeze()