import json
import shutil
import sys
from argparse import ArgumentParser
# from allennlp.commands import main
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_module_and_submodules
from allennlp.common import Registrable, Params
from allennlp.models import Model

def recursive_update(original, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            recursive_update(original[key], value)
        else:
            original[key] = value
    return original

def load_defaults(config_file, keys_to_init):
    with open(config_file, "r") as f:
        config = json.load(f)
    defaults = {}
    for key in keys_to_init:
        if key in config:
            defaults[key] = config[key]
        else:
            defaults[key] = {}
    return defaults

if __name__ == "__main__":
    # 模块加载
    import_module_and_submodules("chunk")

    # 参数解析
    parser = ArgumentParser()
    parser.add_argument("--config", default="", help="input config json file")
    parser.add_argument("--model", default="", help="model output directory")
    parser.add_argument("--train_data", default="", help="training file")
    parser.add_argument("--eval_data", default="", help="evaluation file")
    parser.add_argument("--epoch", type=int, default=0, help="number of epochs")
    parser.add_argument("--batch", type=int, default=0, help="batch size")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device")
    args = parser.parse_args()

    # 配置文件路径
    config_file = args.config if args.config else "config/chunk_oia.json"
    serialization_dir = args.model if args.model else "trained_model/test_for_fun"

    # 在调试模式下删除输出目录
    if not args.model or not args.config:
        shutil.rmtree(serialization_dir, ignore_errors=True)

    # 初始化overrideD的默认值
    keys_to_init = ["data_loader", "trainer", "model"]
    overrideD = load_defaults(config_file, keys_to_init)

    # 根据输入参数递归更新overrideD
    overrideD["trainer"]["cuda_device"] = args.cuda
    if args.train_data and args.eval_data:
        overrideD["train_data_path"] = args.train_data
        overrideD["validation_data_path"] = args.eval_data

    overrideD.setdefault("model", {}).setdefault("tuple_metric", {})
    overrideD["model"]["tuple_metric"]["output_path"] = serialization_dir

    if args.epoch > 0:
        overrideD["trainer"]["num_epochs"] = args.epoch
    if args.batch > 0:
        overrideD["data_loader"]["batch_size"] = args.batch

    # 将overrideD转为JSON字符串
    overrides = json.dumps(overrideD)
    
    params = Params.from_file(config_file, overrides)
    
    train_model_from_file(parameter_filename=config_file,
                          serialization_dir=serialization_dir,
                          recover=False,
                          overrides=overrides)

# if __name__ == '__main__':
#     import_module_and_submodules('chunk')
#     # print(Model.list_available())
    
#     parser = ArgumentParser()
#     parser.add_argument("--config", default='', help="input config json file")
#     parser.add_argument("--model", default='', help="model output directory")
#     parser.add_argument("--train_data", default='', help="training file")
#     parser.add_argument("--eval_data", default='', help="evaluation file")
#     parser.add_argument("--epoch", type=int, default=0, help="number of epoches")
#     parser.add_argument("--batch", type=int, default=0, help="batch size")
#     parser.add_argument("--cuda", type=int, default=0, help="batch size")
#     args = parser.parse_args()

#     overrideD = dict()
#     # overrideD['iterator'] = dict()
#     overrideD['data_loader'] = dict()
#     overrideD['trainer'] = dict()
#     overrideD['model'] = dict()
#     overrideD['trainer']["cuda_device"] = args.cuda

#     if args.train_data != '' and args.eval_data != '':
#         overrideD['train_data_path'] = args.train_data
#         overrideD['validation_data_path'] = args.eval_data

#     # this section is used for debugging
#     if args.model == '' or args.config == '':
#         config_file = "config/chunk_oia.json"
#         serialization_dir = "trained_model/test_for_fun"
#         # erase directory, only applicable to deugging
#         shutil.rmtree(serialization_dir, ignore_errors=True)
#     else:
#         serialization_dir = args.model
#         config_file = args.config

#     # writing predictions to output folders:
#     overrideD['model']["tuple_metric"] = dict()
#     overrideD['model']["tuple_metric"]["output_path"] = serialization_dir

#     if args.epoch > 0:
#         overrideD['trainer']["num_epochs"] = args.epoch
#     if args.batch > 0:
#         # overrideD['iterator']["batch_size"] = args.batch
#         overrideD['data_loader']["batch_size"] = args.batch

#     overrides = json.dumps(overrideD)

#     params = Params.from_file(config_file, overrides)
#     print('params', params)
    
#     train_model_from_file(parameter_filename=config_file,
#                           serialization_dir=serialization_dir,
#                           recover=False,
#                           overrides=overrides)


