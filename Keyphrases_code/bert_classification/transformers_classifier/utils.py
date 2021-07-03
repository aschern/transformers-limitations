import os
from transformers import DataProcessor, InputExample
from sklearn.metrics import f1_score, mean_absolute_error
import numpy as np


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1_macro(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2
    }


def acc_and_f1_macro_mae(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "MAE": mean_absolute_error(y_true=np.array(labels, dtype=np.int), y_pred=np.array(preds, dtype=np.int))
    }


def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ("ra", "ra_neutral", "keyphrases"):
        return acc_and_f1_macro(preds, labels)
    elif task_name in ("quotes", "commas", "length"):
        return acc_and_f1_macro_mae(preds, labels)
    else:
        raise KeyError(task_name)


class RAProcessor(DataProcessor):
    def get_train_examples(self, file_path):
        """See base class."""
        return self._create_examples(self._read_tsv(file_path), "train")

    def get_dev_examples(self, file_path):
        """See base class."""
        return self._create_examples(self._read_tsv(file_path), "dev_matched")

    def get_test_examples(self, file_path):
        """See base class."""
        return self._create_examples(self._read_tsv(file_path), "test")

    def get_labels(self):
        """See base class."""
        return ['DIFF', 'SAME']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 or line == []:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]

            if len(line) < 3:
                label = self.get_labels()[0]
            else:
                label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    

class RANeutralProcessor(RAProcessor):
    def get_labels(self):
        return ['DIFF', 'SAME', 'NEUT']


class LengthProcessor(DataProcessor):
    def get_train_examples(self, file_path):
        """See base class."""
        return self._create_examples(self._read_tsv(file_path), "train")

    def get_dev_examples(self, file_path):
        """See base class."""
        return self._create_examples(self._read_tsv(file_path), "dev_matched")

    def get_test_examples(self, file_path):
        """See base class."""
        return self._create_examples(self._read_tsv(file_path), "test")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(63)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 or line == []:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]

            if len(line) < 2:
                label = self.get_labels()[0]
            else:
                label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples
    
class CommasProcessor(LengthProcessor):
    def get_labels(self):
        return [str(i) for i in range(6)]
    
    
class QuotesProcessor(LengthProcessor):
    def get_labels(self):
        return [str(i) for i in range(37)]

    
class KeyphrasesProcessor(RAProcessor):
    def get_labels(self):
        return ['Task', 'Material', 'Process']


glue_processors = {
    "ra": RAProcessor,
    "ra_neutral": RANeutralProcessor, 
    "length": LengthProcessor,
    "commas": CommasProcessor,
    "quotes": QuotesProcessor,
    "keyphrases": KeyphrasesProcessor
}


glue_tasks_num_labels = {
    "ra": 2,
    "ra_neutral": 3,
    "length": 63,
    "commas": 6,
    "quotes": 37,
    "keyphrases": 3
}


glue_output_modes = {
    "ra": "classification",
    "ra_neutral": "classification",
    "length": "classification",
    "commas": "classification",
    "quotes": "classification",
    "keyphrases": "classification"
}
