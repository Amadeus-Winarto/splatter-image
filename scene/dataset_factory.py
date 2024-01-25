from .srn import SRNDataset
from .co3d import CO3DDataset
from .mixamo import MixamoDataset

def get_dataset(cfg, name):
    if cfg.data.category == "cars" or cfg.data.category == "chairs":
        return SRNDataset(cfg, name)
    elif cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
        return CO3DDataset(cfg, name)
    elif cfg.data.category == "mixamo":
        return MixamoDataset(cfg, name)
    else: 
        raise NotImplementedError