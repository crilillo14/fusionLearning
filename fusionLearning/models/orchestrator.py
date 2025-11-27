"""

Mass train, validate, test, and infer models.

Usage: python orchestrator.py <path to json of workorders>

Default payload path: fusionLearning/models/payload.json
"""

from fusionLearning.models.distributed import launch_training, available_encoder_types, arch_dict
import sys
import os
import json

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        models_to_train_path = os.path.join(os.path.dirname(__file__), "payload.json")
    else:
        models_to_train_path = sys.argv[1]
    
    models_to_train = json.load(open(models_to_train_path, "r"))
    
    for model_to_train in models_to_train:
        
        assert model_to_train["dataset"] in ["CUB", "Cityscapes"]
        assert model_to_train["arch_name"] in arch_dict.values()
        assert model_to_train["encoder"] in available_encoder_types.values()
        
        launch_training(model_to_train["dataset"], model_to_train["arch_name"], model_to_train["encoder"]) 
    
            


    
    