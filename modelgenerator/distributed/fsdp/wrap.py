import inspect
from typing import Iterable, Set, Type

from modelgenerator.backbones.base import SequenceBackboneInterface

import torch.nn as nn
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, CustomPolicy


def get_class_by_path(class_path):
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class


def peft_wrap_policy_fn(module: nn.Module) -> bool:
    """Lambda function for CustomPolicy that wraps frozen moduels separately."""
    if (
        len(list(module.named_children())) == 0
        and getattr(module, "weight", None) is not None
        and module.weight.requires_grad
    ):
        return True
    return False


class AutoWrapPolicy(ModuleWrapPolicy):
    """Set optmized default fsdp wrap policies for given backbones.

    Args:
        backbone_classes (Iterable[Type[SequenceBackboneInterface]] | Type[SequenceBackboneInterface]):
            backebone classes
        use_peft (bool): whether the backbones use peft
        wrap_modules (Optional[Set[Type[nn.Module]]]): manually overwrite modules to wrap
    """

    def __init__(
        self,
        backbone_classes: (
            Iterable[Type[SequenceBackboneInterface]] | Type[SequenceBackboneInterface]
        ) = [],
        use_peft: bool = False,
        wrap_modules: Set[Type[nn.Module]] | None = None,
    ):
        if wrap_modules:
            module_classes_set = wrap_modules
        else:
            if inspect.isclass(backbone_classes):
                backbone_classes = [backbone_classes]
            module_classes_set = set()
            for backbone in backbone_classes:
                for import_path in backbone.fsdp_wrap_modules:
                    module_classes_set.add(get_class_by_path(import_path))
        super().__init__(module_classes=module_classes_set)
        self.use_peft = use_peft

        def lambda_fn(module):
            module_classes = tuple(self._module_classes)
            return (self.use_peft and peft_wrap_policy_fn(module)) or isinstance(
                module, module_classes
            )

        self._policy = CustomPolicy(lambda_fn)

    def _run_policy(self, root_module, ignored_modules, root_kwargs):
        return self._policy._run_policy(root_module, ignored_modules, root_kwargs)

    def __call__(self, module, recurse, *args, **kwargs):
        return super().__call__(
            module, recurse, *args, **kwargs
        ) or self._policy._lambda_fn(module)
