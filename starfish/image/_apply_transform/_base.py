from abc import abstractmethod
from typing import Type

from starfish.image._learn_transform.transforms_list import TransformsList
from starfish.imagestack.imagestack import ImageStack
from starfish.pipeline import PipelineComponent
from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.util import click


COMPONENT_NAME = "apply_transform"


class ApplyTransform(PipelineComponent):

    @classmethod
    def pipeline_component_type_name(cls) -> str:
        return COMPONENT_NAME

    @classmethod
    def _cli_run(cls, ctx, instance):
        output = ctx.obj["output"]
        stack = ctx.obj["stack"]
        transforms_list = ctx["transforms_list"]
        transformed = instance.run(stack, transforms_list)
        transformed.export(output)

    @staticmethod
    @click.group("apply_transform")
    @click.option("-i", "--input", type=click.Path(exists=True))
    @click.option("-o", "--output", required=True)
    @click.option("--transformation-list", required=True, type=click.Path(exists=True),
                  help="The list of transformations to apply to the ImageStack.")
    @click.pass_context
    def _cli(ctx, input, output, transforms_list):
        print("Applying Transform to images...")
        ctx.obj = dict(
            component=ApplyTransform,
            input=input,
            output=output,
            stack=ImageStack.from_path_or_url(input),
            transforms_list=TransformsList.from_json(transforms_list)
        )


class ApplyTransformBase(AlgorithmBase):
    @classmethod
    def get_pipeline_component_class(cls) -> Type[PipelineComponent]:
        return ApplyTransform

    @abstractmethod
    def run(self, stack, transforms_list) -> ImageStack:
        """Performs registration on the stack provided."""
        raise NotImplementedError()
