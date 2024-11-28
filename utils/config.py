import yaml
from easydict import EasyDict
import os
from .logger import print_log

def count_trainable_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)

def model_summary(model):
    print("\n=== Model Summary ===")
    print(f"Total parameters: {count_trainable_parameters(model.parameters())}")
    
    print("\n=== Module-wise Summary ===")
    for name, module in model.named_children():
        num_params = count_trainable_parameters(module.parameters())
        print(f"Module: {name}, Trainable Parameters: {num_params}")
        
    print("\n=== Parameter-wise Details ===")
    for name, param in model.named_parameters():
        param_type = "Trainable" if param.requires_grad else "Frozen"
        print(f"Param: {name}, Shape: {param.shape}, {param_type}")

def model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_bytes = total_params * 4  # Assuming float32
    model_size_mb = param_size_bytes / (1024 ** 2)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print("done")

def get_instance(config, available_classes, addictional_params={}):
    cls = available_classes[config['type']]
    params = config.get('kwargs', {})
    return cls(**params, **addictional_params)


def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger = logger)

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger = logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger = logger)

def merge_new_config(config, new_config, root=''):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                config_path = root + new_config['_base_']
                # print(f"config path: {config_path}")
                with open(config_path, 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val, root)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val, root)
    return config

def cfg_from_yaml_file(cfg_file, merge=True, root=''):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    if merge: merge_new_config(config=config, new_config=new_config, root=root)  
    else: config = new_config      
    return config

def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger = logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger = logger)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    return config

def save_experiment_config(args, config, logger = None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))
    print_log(f'Copy the Config file from {args.config} to {config_path}',logger = logger )