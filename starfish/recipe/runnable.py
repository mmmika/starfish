import inspect
import warnings
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set,
    Type,
)

from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent, PipelineComponentType
from .errors import (
    ConstructorError,
    ConstructorExtraParameterWarning,
    ExecutionError,
    TypeInferenceError,
)
from .filesystem import FileProvider, TypedFileProvider


class Runnable:
    """Runnable represents a single invocation of a pipeline component, with a specific algorithm
    implementation.  For arguments to the algorithm's constructor and run method, it can accept
    :py:class:starfish.recipe.filesystem.FileProvider objects, which represent a file path or url.
    For arguments to the algorithm's run method, it can accept the results of other Runnables.

    One can compose any starfish pipeline run using a directed acyclic graph of Runnables objects.
    """
    def __init__(
            self,
            pipeline_component_name: str,
            algorithm_name: str,
            *inputs,
            **algorithm_options
    ) -> None:
        self._pipeline_component_name = pipeline_component_name
        self._algorithm_name = algorithm_name
        self._raw_inputs = inputs
        self._raw_algorithm_options = algorithm_options

        self._pipeline_component_cls: Type[PipelineComponent] = \
            PipelineComponentType.get_pipeline_component_type_by_name(self._pipeline_component_name)
        self._algorithm_cls: Type = getattr(
            self._pipeline_component_cls, self._algorithm_name)

        # retrieve the actual __init__ method
        signature = Runnable._get_actual_method_signature(
            self._algorithm_cls.__init__)
        assert next(iter(signature.parameters.keys())) == "self"

        formatted_algorithm_options: MutableMapping[str, Any] = {}
        for algorithm_option_name, algorithm_option_value in self._raw_algorithm_options.items():
            if isinstance(algorithm_option_value, FileProvider):
                try:
                    option_class = signature.parameters[algorithm_option_name].annotation
                except KeyError:
                    warnings.warn(
                        f"Constructor for {str(self)} does not have an explicitly typed parameter "
                        + f"{algorithm_option_name}.",
                        category=ConstructorExtraParameterWarning,
                    )
                    continue
                try:
                    provider = TypedFileProvider(algorithm_option_value, option_class)
                except TypeError as ex:
                    raise TypeInferenceError(
                        f"Error inferring the types for the parameters to the algorithm's"
                        + f" constructor for {str(self)}") from ex
                formatted_algorithm_options[algorithm_option_name] = provider.load()
            else:
                formatted_algorithm_options[algorithm_option_name] = algorithm_option_value

        try:
            self._algorithm_instance: AlgorithmBase = self._algorithm_cls(
                **formatted_algorithm_options)
        except Exception as ex:
            raise ConstructorError(f"Error instantiating the algorithm for {str(self)}") from ex

        # retrieve the actual run method
        signature = Runnable._get_actual_method_signature(
            self._algorithm_instance.run)  # type: ignore
        keys = list(signature.parameters.keys())

        assert keys[0] == "self"
        keys = keys[1:]  # ignore the "self" parameter
        assert len(self._raw_inputs) <= len(keys)

        formatted_inputs: MutableSequence = []
        for _input, key in zip(self._raw_inputs, keys):
            if isinstance(_input, FileProvider):
                annotation = signature.parameters[key].annotation
                try:
                    provider = TypedFileProvider(_input, annotation)
                except TypeError as ex:
                    raise TypeInferenceError(
                        f"Error inferring the types for the parameters to the algorithm's"
                        + f" run method for {str(self)}") from ex
                formatted_inputs.append(provider)
            else:
                formatted_inputs.append(_input)
        self._inputs: Sequence = formatted_inputs

    @staticmethod
    def _get_actual_method_signature(run_method: Callable) -> inspect.Signature:
        if hasattr(run_method, "__closure__"):
            # it's a closure, probably because of AlgorithmBaseType.run_with_logging.  Unwrap to
            # find the underlying method.
            closure = run_method.__closure__  # type: ignore
            if closure is not None:
                run_method = closure[0].cell_contents

        return inspect.signature(run_method)

    @property
    def runnable_dependencies(self) -> Set["Runnable"]:
        """Retrieves a set of Runnables that this Runnable depends on."""
        return set(runnable for runnable in self._inputs if isinstance(runnable, Runnable))

    def run(self, previous_results: Mapping["Runnable", Any]) -> Any:
        """Do the heavy computation involved in this runnable."""
        inputs = list()
        for _input in self._inputs:
            if isinstance(_input, Runnable):
                inputs.append(previous_results[_input])
            elif isinstance(_input, TypedFileProvider):
                inputs.append(_input.load())
            else:
                inputs.append(_input)
        try:
            return self._algorithm_instance.run(*inputs)  # type: ignore
        except Exception as ex:
            raise ExecutionError(f"Error running the algorithm for {str(self)}") from ex

    def __str__(self):
        inputs_arr = [""]
        inputs_arr.extend([str(raw_input) for raw_input in self._raw_inputs])
        algorithm_options_arr = [""]
        algorithm_options_arr.extend([
            f"{algorithm_option_name}={str(algorithm_option_value)}"
            for algorithm_option_name, algorithm_option_value in
            self._raw_algorithm_options.items()
        ])

        inputs_str = ", ".join(inputs_arr)
        algorithm_options_str = ", ".join(algorithm_options_arr)

        return (f"compute("
                + f"\"{self._pipeline_component_name}\","
                + f" \"{self._algorithm_name}\""
                + f"{inputs_str}"
                + f"{algorithm_options_str})")
