import argparse
import json
import os

from runner.runner import Runner

def copyFile(runner:Runner, target_path='./result', save_interval = 10):
    import shutil
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for epoch in range(runner.start_epoch + save_interval - runner.start_epoch%save_interval, 2001, save_interval):
        model_file_name = runner.getModelFilename(epoch)
        shutil.copyfile(os.path.join(runner.model_save_path, model_file_name), os.path.join(target_path,model_file_name))
    

def train_model(config, model_step):
    runner = Runner(config, model_step= model_step)
    max_epoch = runner.config_model['epochs']

    while runner.cur_epoch != max_epoch:
        runner.train()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train start")
    parser.add_argument("--config", type=str, default="./config/config.json")
    args = parser.parse_args()

    json_file = args.config
    with open(json_file, "r") as f:
        config = json.load(f)

    config['data']['num_bones'] = 20

    print("----------------------train step1---------------------------")
    train_model(config, model_step = 1)
    print("----------------------train step2---------------------------")
    train_model(config, model_step = 2)
