from typing import Generator
from pipeline.pipeline import Step
from helpers import read_conllu_file, read_txt_file


class LoadConlluDataStep(Step):
    """
    Step to load data from a .conllu file
    """
    def __init__(self, filenames: list, batch_size: int):
        self.filenames = filenames
        self.batch_size = batch_size

    def run(self, batches: Generator) -> Generator:
        """
        Runs the step
        """
        return read_conllu_file(self.filenames, self.batch_size)


class LoadTxtDataStep(Step):
    """
    Step to load data from a txt file, with one row per sentence
    """

    def __init__(self, filename: str):
        self.filename = filename

    def run(self, batches: Generator) -> Generator:
        """
        Runs the step
        """
        return read_txt_file(self.filename)
