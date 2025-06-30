


def get_paths() -> list[str]:
    """
    Get a list of paths to all .pth files in the models directory.
    """
    os.chdir("fusionLearning/models")

    
    

all_pth_paths = list(get_paths())

def get_base_models(models : list[str],
                    get_all : bool = True,
                    num_models : int = 5):
    """
    Get a list of base models to use for ensembling.
    
    Args:
        models: List of model names to use.
        get_all: Whether to get all models or not.
        num_models: Number of models to get.
    """

    if get_all:
        return all_pth_paths
    else:
        return random.sample(all_pth_paths, num_models)

class ModelPathIterator:
    """
    A simple iterator for the model pths, returning their filepaths.
    """
    def __init__(self, models):
        self.models = models
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.models):
            path = self.models[self.index]
            self.index += 1
            return path
        raise StopIteration

def get_model_path_iterator(models : list[str],
                            get_all : bool = True,
                            num_models : int = 5):
    """
    Get a ModelPathIterator for the base models to use for ensembling.
    
    Args:
        models: List of model names to use.
        get_all: Whether to get all models or not.
        num_models: Number of models to get.
    """
    return ModelPathIterator(get_base_models(models, get_all, num_models))
