import os
from pyrouge import Rouge155
from shutil import rmtree


class RougeManagement():
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._system_dir = os.path.join(self._root_dir, 'system')
        self._model_dir = os.path.join(self._root_dir, 'model')
        if not os.path.exists(self._root_dir):
            os.mkdir(self._root_dir)
        if os.path.exists(self._system_dir):
            rmtree(self._system_dir)
        os.mkdir(self._system_dir)
        if os.path.exists(self._model_dir):
            rmtree(self._model_dir)
        os.mkdir(self._model_dir)
        self.cur_id = 0

    def add_pair(self, system, model):
        self._write_sentences(
            system,
            os.path.join(self._system_dir,
                         "system.%03d.txt" % self.cur_id))
        self._write_sentences(
            model,
            os.path.join(self._model_dir,
                         "model.%03d.txt" % self.cur_id))
        self.cur_id += 1

    def _write_sentences(self, sentences, file_name):
        with open(file_name, 'w') as f:
            for sentence in sentences:
                print(sentence, file=f)

    def evaluate(self):
        self.rouge = Rouge155()
        self.rouge.system_dir = self._system_dir
        self.rouge.model_dir = self._model_dir
        self.rouge.system_filename_pattern = 'system.(\d+).txt'
        self.rouge.model_filename_pattern = 'model.#ID#.txt'
        output = self.rouge.convert_and_evaluate()
        return output
    
    def output_to_dict(self, output):
        return self.rouge.output_to_dict(output)

    def close(self):
        rmtree(self._root_dir)
