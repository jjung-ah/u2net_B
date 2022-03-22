
import logging
from datetime import timedelta
import torch
import torch.multiprocessing as mp
# import torch.distributions as dist
import torch.distributed as dist
from utils import comm

DEFAULT_TIMEOUT = timedelta(minutes=30)

def launch(
        main_func,
        num_gpus_per_machine,
        num_machines=1,
        machine_rank=0,
        dist_url=None,
        cfg=None,
        timeout=DEFAULT_TIMEOUT
):
    """

    Args:
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): total number of machine
        machine_rank (int):
        dist_url (str): url to connect to for distributed jobs,
            e.g. "tcp://127.0.0.1:8686"
        args (tuple): arguments passed to main_func

    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                cfg,
                timeout
            ),
            daemon=False,
        )
    else:
        main_func(cfg)



def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    cfg,
    timeout
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(name=cfg.logs.name)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    main_func(cfg)
