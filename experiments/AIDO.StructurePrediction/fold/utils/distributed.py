

import os

import torch


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class DistWrapper:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.num_nodes = int(self.world_size // self.local_world_size)
        self.node_rank = int(self.rank // self.local_world_size)

    def all_gather_object(self, obj, group=None):
        """Function to gather objects from several distributed processes.
        It is now only used by sync metrics in logger due to security reason.
        """
        if self.world_size > 1 and distributed_available():
            with torch.no_grad():
                obj_list = [None for _ in range(self.world_size)]
                torch.distributed.all_gather_object(obj_list, obj, group=group)
                return obj_list
        else:
            return [obj]


DIST_WRAPPER = DistWrapper()


def traverse_and_aggregate(dict_list, aggregation_func=None):
    """Traverse list of dicts and merge into a single dict with leaf values joined to list."""
    merged_dict = {}
    all_keys = set().union(*dict_list)
    for key in all_keys:
        agg_value = [m[key] for m in dict_list if key in m]

        if isinstance(agg_value[0], dict):
            merged_dict[key] = traverse_and_aggregate(
                agg_value, aggregation_func=aggregation_func
            )
        else:
            if aggregation_func is not None:
                agg_value = aggregation_func(agg_value)
            merged_dict[key] = agg_value

    return merged_dict

