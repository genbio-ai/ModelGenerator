import os
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from modelgenerator.tasks import *
from modelgenerator.backbones import *
from modelgenerator.adapters import *
from modelgenerator.data import *


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
        parser.link_arguments(
            "model.init_args.backbone.class_path",
            "trainer.strategy.init_args.auto_wrap_policy.init_args.backbone_classes",
        )
        parser.link_arguments(
            "model.init_args.backbone.class_path",
            "data.init_args.backbone_class_path",
        )
        parser.link_arguments(
            "model.init_args.backbone.init_args.use_peft",
            "trainer.strategy.init_args.auto_wrap_policy.init_args.use_peft",
        )
        parser.link_arguments(
            "data",
            "model.init_args.data_module",
            apply_on="instantiate",
        )

    def after_instantiate_classes(self):
        super().after_instantiate_classes()
        # First data compatibility check.
        # It is the earliest point to perform this check, avoiding
        # overheads such as `datamodule.setup` and spawning of distributed training processes.
        # The check is only performed when using the `mgen` CLI. To ensure compatibility is always
        # verified, a second compatibility check is implemented as a callback in the tasks.
        self.model.check_data_compatibility(self.subcommand)


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage) -> None:
        if isinstance(trainer.logger, WandbLogger):
            self.parser.save(
                self.config,
                os.path.join(trainer.logger.experiment.dir, "run-config.yaml"),
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )
            trainer.logger.experiment.config.update(self.config.as_dict())


def cli_main():
    """
    Entrypoint for mgen command
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    default_config = os.path.join(base_dir, "configs/defaults.yaml")
    try:
        MyLightningCLI(
            save_config_kwargs={"overwrite": True},
            parser_kwargs={"fit": {"default_config_files": [default_config]}},
            auto_configure_optimizers=False,
            save_config_callback=LoggerSaveConfigCallback,
        )
    except (IndexError, StopIteration) as e:
        from importlib.metadata import version as get_version
        from packaging.version import Version

        parser_version = Version(get_version("jsonargparse"))
        if parser_version <= Version("4.36.0"):
            print(
                "With jsonargparse<=4.36.0, "
                "strategies other than FSDP have to be specified using "
                "their string names instead of class paths. For example, use "
                "strategy: ddp or strategy: ddp_find_unused_parameters_true. "
                "Upgrade jsonargparse to eliminate this limitation."
            )
            return 1
        else:
            raise e


if __name__ == "__main__":
    cli_main()
