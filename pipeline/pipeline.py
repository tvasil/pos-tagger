from typing import Generator


class Step:
    """
    Defines a step in the pipeline
    """

    def run(self, batches: Generator) -> Generator:
        """
        Run this step though the generator and yield any needed content
        """
        raise NotImplementedError("Subclasses must implement this method")


class Pipeline:
    """
    At its core, a pipeline is just a list of steps
    """

    def __init__(self, steps: list):
        self.steps = steps

    def run(self):
        def empty_gen():
            yield from ()

        retval = empty_gen()
        for step in self.steps:
            retval = step.run(retval)
