
from utils.comm import get_world_size
from .datasets.build import build_train_val_datasets, build_test_datasets
import torch.utils.data as torchdata
from torch.utils.data.distributed import DistributedSampler

def build_batch_data_loader(
        dataset,
        total_batch_size,
        num_workers=0,
        is_shuffle=False,
        collate_fn=None,
        drop_last = True,
        pin_memory=True
):
    world_size = get_world_size()
    assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    sampler = None if world_size == 1 else DistributedSampler(dataset)

    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler
    )

def build_segmentation_loader(
        cfg,
        datasets,
        drop_last=False,
        pin_memory=True
):
    datasets = build_train_val_datasets(cfg, datasets)
    return build_batch_data_loader(
        datasets,
        cfg.training.solver.image_per_batch,
        cfg.training.data_loader.num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory
    )
#
# def build_segmentation_val_loader(
#         cfg,
#         datasets,
#         drop_last=False,
#         pin_memory=True
# ):
#     datasets = build_datasets(cfg, datasets)
#     return build_batch_data_loader(
#         datasets,
#         cfg.training.solver.image_per_batch,
#         cfg.training.data_loader.num_workers,
#         drop_last=drop_last,
#         pin_memory=pin_memory
#     )


def build_segmentation_test_loader(
        cfg,
        datasets,
        drop_last=False,
        pin_memory=True
):
    datasets = build_test_datasets(cfg, datasets)
    return build_batch_data_loader(
        datasets,
        cfg.training.solver.image_per_batch,
        cfg.training.data_loader.num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory
    )