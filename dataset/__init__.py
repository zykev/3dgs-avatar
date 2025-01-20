from .zjumocap import ZJUMoCapDataset
from .people_snapshot import PeopleSnapshotDataset
from .refined_zjumocap import RefinedZJUMoCapDataset

def load_dataset(cfg, split='train'):
    dataset_dict = {
        'zjumocap': ZJUMoCapDataset,
        'refined_zjumocap': RefinedZJUMoCapDataset,
        'people_snapshot': PeopleSnapshotDataset,
    }
    return dataset_dict[cfg.name](cfg, split)
