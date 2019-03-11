from typing import Type

from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import AlgorithmBase, import_all_submodules, PipelineComponent
from starfish.util import click
from . import _base
import_all_submodules(__file__, __package__)


class LearnTransform(PipelineComponent):

    @classmethod
    def _get_algorithm_base_class(cls) -> Type[AlgorithmBase]:
        return _base.LearnTransformBase

    @classmethod
    def _cli_run(cls, ctx, instance, *args, **kwargs):
        # TODO FIX THIS OUTPUT SHOULD BE GEOMETRIC TRANSFORM
        output = ctx.obj["output"]
        stack = ctx.obj["stack"]
        transformed = instance.run(stack)
        transformed.export(output)

    @staticmethod
    @click.group("apply_transform")
    @click.option("-i", "--input", type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.pass_context
    def _cli(ctx, input, output):
        """smooth, sharpen, denoise, etc"""
        print("Applying Transform to images...")
        ctx.obj = dict(
            component=LearnTransform,
            input=input,
            output=output,
            stack=ImageStack.from_path_or_url(input),
        )
