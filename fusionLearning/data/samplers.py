from fusionLearning.data.dataloaders import CUBDataset
from pytorch.utils.dataLoader import DataLoader
from torch.nn.utils import DistributedSampler
from dataclasses import dataclass

@dataclass
class SamplerConfig:
    num_workers : int
    processes_per_node : int
    world_size : int 

def create_samplers(loaders : list[DataLoader], config : SamplerConfig):
    samplers = {}
    for loader in loaders:
        sampler = DistributedSampler(loader)
        samplers[loader] = sampler
        
    return samplers
