import argparse
import json
import os
from runner.runner import Runner
from model.utils.model import loadCheckPoint

def predict(config,save_post_fix = ""):
    des_path = config['model_save_path'] 
    config['data']['train_folders'] =   [config['data']['train_folders'][0]]

    runner = Runner(config, model_step=2,mode ='predict')
    coarse_checkpoint = loadCheckPoint(os.path.join(des_path,runner.getModelFilename(150,model_step=1)),device=runner.device)
    runner.c_model.load_state_dict(coarse_checkpoint['model_state_dict'])
    
    runner.train_data.cloth.UpdateBoneWeights(coarse_checkpoint['weights'].to('cpu'))
    runner.lbs.setWeigths(runner.train_data.cloth.mixed_weights.to(runner.device))

    fine_checkpoint = loadCheckPoint(os.path.join(des_path,runner.getModelFilename(150,model_step=2)),device=runner.device)
    runner.f_model.load_state_dict(fine_checkpoint['model_state_dict'])

    runner.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train start")
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument("--folder", type=str, default="Motion0")
    args = parser.parse_args()

    json_file = args.config
    with open(json_file, "r") as f:
        config = json.load(f)

    config['data']['test_folders'] = [args.folder]

    predict(config)
