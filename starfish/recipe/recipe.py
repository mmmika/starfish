from typing import (
    AbstractSet,
    Any,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set,
)

from .filesystem import FileProvider, FileTypes
from .runnable import Runnable


class Execution:
    """Encompasses the state of a single execution of a recipe."""
    def __init__(
            self,
            runnable_sequence: Sequence[Runnable],
            outputs: Sequence[Runnable],
            output_paths: Sequence[str],
    ) -> None:
        self._runnable_sequence = iter(runnable_sequence)
        self._outputs = outputs
        self._output_paths = output_paths

        # build a set of runnables whose outputs are saved to disk.  These outputs must never be
        # dropped.
        self._output_runnables: AbstractSet[Runnable] = self._outputs

        # build a set of all the runnables we need to arrive at the outputs, and a mapping from each
        # runnable to its dependents.
        needed_runnables: Set[Runnable] = set()
        runnable_dependents: MutableMapping[Runnable, Set[Runnable]] = dict()
        for output_runnable in self._outputs:
            Execution.build_graph(output_runnable, needed_runnables, runnable_dependents)
        self.needed_runnables: AbstractSet[Runnable] = needed_runnables
        self.runnable_dependents: Mapping[Runnable, AbstractSet[Runnable]] = runnable_dependents

        # completed results
        self._completed_runnables: Set[Runnable] = set()
        self._completed_results: MutableMapping[Runnable, Any] = dict()
        self._incomplete_outputs: Set[Runnable] = set(self._output_runnables)

    @property
    def complete(self) -> bool:
        return len(self._incomplete_outputs) == 0

    def run_one_tick(self) -> None:
        """Run one tick of the execution graph."""
        while True:
            candidate_runnable = next(self._runnable_sequence)

            if candidate_runnable in self.needed_runnables:
                break

        result = candidate_runnable.run(self._completed_results)

        # update what's been done, and try to remove ourselves from the final outputs, if
        # applicable.
        self._completed_runnables.add(candidate_runnable)
        self._completed_results[candidate_runnable] = result
        self._incomplete_outputs.discard(candidate_runnable)

        # examine all the dependencies, and discard the results if no one else needs it.
        for dependency in candidate_runnable.runnable_dependencies:
            if dependency in self._output_runnables:
                # it's required by the outputs, so preserve this.
                continue

            for dependent in self.runnable_dependents[dependency]:
                if dependent not in self._completed_runnables:
                    # someone still needs this runnable's result.
                    break
            else:
                # every dependent is complete.  drop the result.
                del self._completed_results[dependency]

    def save(self) -> None:
        assert self.complete
        # all the outputs should be satisfied.
        for runnable, output_path in zip(self._output_runnables, self._output_paths):
            # get the result
            result = self._completed_results[runnable]

            filetype = FileTypes.resolve_by_instance(result)
            filetype.save(result, output_path)

    def run_and_save(self) -> None:
        while not self.complete:
            self.run_one_tick()
        self.save()

    @staticmethod
    def build_graph(
            runnable: Runnable,
            needed_runnables: Set[Runnable],
            runnable_dependents: MutableMapping[Runnable, Set[Runnable]],
    ) -> None:
        if runnable in needed_runnables:
            return

        needed_runnables.add(runnable)
        for dependency in runnable.runnable_dependencies:
            # mark ourselves a dependent of each of our dependencies.
            if dependency not in runnable_dependents:
                runnable_dependents[dependency] = set()
            runnable_dependents[dependency].add(runnable)
            Execution.build_graph(dependency, needed_runnables, runnable_dependents)


class OrderedSequence:
    def __init__(self) -> None:
        self._sequence: MutableSequence[Runnable] = list()

    def __call__(self, *args, **kwargs):
        result = Runnable(*args, **kwargs)
        self._sequence.append(result)
        return result

    @property
    def sequence(self) -> Sequence[Runnable]:
        return self._sequence


class Recipe:
    def __init__(
            self,
            recipe_str: str,
            input_paths_or_urls: Sequence[str],
            output_paths: Sequence[str],
    ):
        ordered_sequence = OrderedSequence()
        outputs: MutableMapping[int, Runnable] = {}
        recipe_scope = {
            "file_inputs": [
                FileProvider(input_path_or_url)
                for input_path_or_url in input_paths_or_urls
            ],
            "compute": ordered_sequence,
            "file_outputs": outputs,
        }
        ast = compile(recipe_str, "<string>", "exec")
        exec(ast, recipe_scope)

        assert len(outputs) == len(output_paths), \
            "Recipe generates more outputs than output paths provided!"

        # verify that the outputs are sequential.
        ordered_outputs: MutableSequence[Runnable] = list()
        for ix in range(len(outputs)):
            assert ix in outputs, \
                f"file_outputs[{ix}] is not set"
            assert isinstance(outputs[ix], Runnable), \
                f"file_outputs[{ix}] is not the result of a compute(..)"
            ordered_outputs.append(outputs[ix])

        self._runnable_order = ordered_sequence.sequence
        self._outputs: Sequence[Runnable] = ordered_outputs
        self._output_paths = output_paths

    def execution(self) -> Execution:
        return Execution(self._runnable_order, self._outputs, self._output_paths)
