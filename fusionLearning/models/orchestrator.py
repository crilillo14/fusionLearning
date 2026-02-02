"""

Mass train, validate, test, and infer models.

Usage: python orchestrator.py <path to json of workorders>

Default payload path: fusionLearning/models/payload.json
"""



import os
import sys

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

from fusionLearning.models.distributed import launch_training, available_encoder_types, arch_dict, flat_encoders
import sys
import os
import json
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Mass training of all available arch-encoder pairings')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--all', action='store_true', help="Train all arch-encoder pairings for the specified dataset.")
    
    args = parser.parse_args()
    
    if args.all:
        for encoder in flat_encoders:
            for arch_name in arch_dict:
                print(f"Training {arch_name} with {encoder} encoder on the {args.dataset} dataset")
                launch_training(arch_name, encoder, args.dataset)
       
       
    """ 
    models_to_train = json.load(open(models_to_train_path, "r"))
    
    for model_to_train in models_to_train:
            
        assert model_to_train["dataset"] in ["CUB", "Cityscapes"]
        assert model_to_train["arch_name"] in arch_dict.keys()
        assert model_to_train["encoder"] in flat_encoders
            
        launch_training(model_to_train["dataset"], model_to_train["arch_name"], model_to_train["encoder"]) 
        
    """
        
            


    
    