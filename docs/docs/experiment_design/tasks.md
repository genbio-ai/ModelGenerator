# Adding Tasks

Tasks are use-cases for pre-trained foundation models.

Pre-trained foundation models (FMs, backbones) improve performance across a wide range of ML tasks.
However, tasks utilize FMs in very different ways, often requiring a unique reimplementation or adaptation for every backbone-task pair, a process that is time-consuming and error-prone.
For FM-enabled research and development to be practical, modularity and reusability are essential.

AIDO.ModelGenerator `tasks` enable rapid prototyping and experimentation through hot-swappable `backbone` and `adapter` components, which make use of standard interfaces.
All of this is made possible by the PyTorch Lightning framework, which provides the [LightningModule](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) interface for hardware-agnostic training, evaluation, and prediction, as well as configified experiment management and extensive [CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) support.


Available Tasks:
Inference, MLM, SequenceClassification, TokenClassification, PairwiseTokenClassification, Diffusion, ConditionalDiffusion, SequenceRegression, Embed


> Note: Adapters and Backbones are typed as [`Callables`](https://jsonargparse.readthedocs.io/en/stable/index.html#callable-type), since some args are reserved to automatically configure the adapter with the backbone.
Create an `AdapterCallable` signature for a task to specify which arguments are configurable, and which are reserved.
>

## Adding Adapters

Adapters serve as a linker between a backbone's output and a task's objective function.

They are simple [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) objects that use the backbone interface to configure their weights and forward pass.
Their construction is handled within the task's `configure_model` method.
Each task only tolerates a specific adapter type, which all adapters for that task must subclass.
See the `SequenceAdapter` type and implemented `LinearCLSAdapter` for `SequenceRegression` as an example below.

::: modelgenerator.tasks.TaskInterface
    handler: python
    options:
      filters:
        - "!^__"
      members:
        - configure_model
        - collate
        - forward
        - evaluate
      show_root_heading: true
      show_source: true

## Examples

::: modelgenerator.tasks.SequenceRegression
    handler: python
    options:
      filters:
        - "!^__"
      members:
        - configure_model
        - collate
        - forward
        - evaluate
      show_root_heading: true
      show_source: true

::: modelgenerator.adapters.SequenceAdapter
    handler: python
    options:
      filters:
        - "!^__"
      show_root_heading: true
      show_source: true

::: modelgenerator.adapters.LinearCLSAdapter
    handler: python
    options:
      filters:
        - "!^__"
      show_root_heading: true
      show_source: true
