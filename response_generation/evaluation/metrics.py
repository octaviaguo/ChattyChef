#!/usr/bin/env python3

import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
import math
from typing import (
    Any,
    Counter as TCounter,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from nltk.translate import bleu_score as nltkbleu
from nltk import word_tokenize

TScalar = Union[int, float, torch.Tensor]
TVector = Union[List[TScalar], torch.Tensor]
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s


@functools.total_ordering  # type: ignore
class Metric(ABC):
    """
    Base class for storing metrics.
    Subclasses should define .value(). Examples are provided for each subclass.
    """

    @property
    def is_global(self) -> bool:
        """
        Indicates whether this metric should be reported globally or per-task.
        """
        return False

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return False

    @abstractmethod
    def value(self) -> float:
        """
        Return the value of the metric as a float.
        """
        pass

    @abstractmethod
    def __add__(self, other: Any):
        raise NotImplementedError

    def __iadd__(self, other):
        return self.__radd__(other)

    def __radd__(self, other: Any):
        if other is None:
            return self
        return self.__add__(other)

    def __str__(self) -> str:
        return f'{self.value():.4g}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value():.4g})'

    def __float__(self) -> float:
        return float(self.value())

    def __int__(self) -> int:
        return int(self.value())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() == other.value()
        else:
            return self.value() == other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() < other.value()
        else:
            return self.value() < other

    def __sub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__sub__ is intentionally limited to floats.')
        return self.value() - other

    def __rsub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.
        NOTE: This is not necessary in python 3.7+.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__rsub__ is intentionally limited to floats.')
        return other - self.value()

    @classmethod
    def as_number(cls, obj: TScalar) -> Union[int, float]:
        if isinstance(obj, torch.Tensor):
            obj_as_number: Union[int, float] = obj.item()
        else:
            obj_as_number = obj  # type: ignore
        assert isinstance(obj_as_number, int) or isinstance(obj_as_number, float)
        return obj_as_number

    @classmethod
    def as_float(cls, obj: TScalar) -> float:
        return float(cls.as_number(obj))

    @classmethod
    def as_int(cls, obj: TScalar) -> int:
        return int(cls.as_number(obj))

    @classmethod
    def many(cls, *objs: List[TVector]):
        """
        Construct many of a Metric from the base parts.
        Useful if you separately compute numerators and denomenators, etc.
        """
        lengths = [len(o) for o in objs]
        objs = list(objs)  # convert from tuple for inplace modification
        for i, o in enumerate(objs):
            if isinstance(o, torch.Tensor):
                # if the tensor is on GPU, make sure we transfer the whole thing
                # at once, instead of one-element-at-a-time during our list
                # comprehension
                objs[i] = o.tolist()
        if len(set(lengths)) != 1:
            raise IndexError(f'Uneven {cls.__name__} constructions: {lengths}')
        return [cls(*items) for items in zip(*objs)]


class AverageMetric(Metric):
    """
    Class that keeps a running average of some metric.
    Examples of AverageMetrics include hits@1, F1, accuracy, etc. These metrics all have
    per-example values that can be directly mapped back to a teacher.
    """

    __slots__ = ('_numer', '_denom')

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, numer: TScalar, denom: TScalar = 1):
        self._numer = self.as_number(numer)
        self._denom = self.as_number(denom)

    def __add__(self, other):
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_numer: TScalar = self._numer + other._numer
        full_denom: TScalar = self._denom + other._denom
        # always keep the same return type
        return type(self)(numer=full_numer, denom=full_denom)

    def value(self) -> float:
        if self._numer == 0 and self._denom == 0:
            # don't nan out if we haven't counted anything
            return 0.0
        if self._denom == 0:
            return float('nan')
        return self._numer / self._denom


class F1Metric(AverageMetric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.
        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values
        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute(
        guess: str, answers: List[str], expose_p_and_r: bool = False
    ):
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        g_tokens = normalize_answer(guess).split()
        scores = [
            F1Metric._prec_recall_f1_score(g_tokens, normalize_answer(a).split())
            for a in answers
        ]
        max_p, max_r, max_f1 = 0, 0, 0
        for p, r, f1 in scores:
            max_p, max_r, max_f1 = max(max_p, p), max(max_r, r), max(f1, max_f1)
        if expose_p_and_r:
            return (F1Metric(max_p, 1), F1Metric(max_r, 1), F1Metric(max_f1, 1))
        else:
            return F1Metric(max_f1, 1)


class BleuMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str], k: int = 4):
        """
        Compute approximate BLEU score between guess and a set of answers.
        """
        try:
            from nltk.translate import bleu_score as nltkbleu
        except ImportError:
            # User doesn't have nltk installed, so we can't use it for bleu
            # We'll just turn off things, but we might want to warn the user
            return None

        # Warning: BLEU calculation *should* include proper tokenization and
        # punctuation etc. We're using the normalize_answer for everything though,
        # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
        # going to be slower than fairseq's (which is written in C), but fairseq's
        # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
        # works with strings, which is better suited for this module.
        weights = [1 / k for _ in range(k)]
        score = nltkbleu.sentence_bleu(
            [normalize_answer(a).split(" ") for a in answers],
            normalize_answer(guess).split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
            weights=weights,
        )

        return BleuMetric(score)

    @staticmethod
    def compute_corpus(guess: List[str], answers: List[List[str]], k: int = 4):
        """
        Compute approximate BLEU score between guess and a set of answers.
        """
        try:
            from nltk.translate import bleu_score as nltkbleu
        except ImportError:
            # User doesn't have nltk installed, so we can't use it for bleu
            # We'll just turn off things, but we might want to warn the user
            return None

        # Warning: BLEU calculation *should* include proper tokenization and
        # punctuation etc. We're using the normalize_answer for everything though,
        # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
        # going to be slower than fairseq's (which is written in C), but fairseq's
        # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
        # works with strings, which is better suited for this module.
        weights = [1 / k for _ in range(k)]
        hypos = [normalize_answer(g).split(" ") for g in guess]

        ref_list = []
        for references in answers:
            tmp = []
            for ref in references:
                tmp.append(ref.split())
            ref_list.append(tmp)

        score = nltkbleu.corpus_bleu(
            ref_list,
            hypos
        )

        return BleuMetric(score)


class JaccardMetric:
    @staticmethod
    def compute_per_pair(text1: str, text2: str, n_gram: int = 1):
        unique_n_gram1 = set()
        unique_n_gram2 = set()
        if n_gram == 1:
            unique_n_gram1 = set(text1)
            unique_n_gram2 = set(text2)
        else:
            for i in range(len(text1) - n_gram):
                unique_n_gram1.add(text1[i:i + n_gram])
            for i in range(len(text2) - n_gram):
                unique_n_gram2.add(text2[i:i + n_gram])
        overlap_n_grams = unique_n_gram1.intersection(unique_n_gram2)
        all_n_grams = unique_n_gram1.union(unique_n_gram2)
        score = len(overlap_n_grams)*1./len(all_n_grams)
        return score

    @staticmethod
    def compute(guess: str, answers: List[str], n_gram: int = 1):
        scores = []
        for answer in answers:
            scores.append(JaccardMetric.compute_per_pair(guess, answer, n_gram))
        return max(scores)


if __name__ == '__main__':
    pred = ["blue is one of three primary colors of pigments in painting and traditional colour theory as well as in rgb colour model"]
    refs = [["blue is my favorite primary color"]]
    print(BleuMetric.compute_corpus(pred, refs).value())

    from datasets import load_dataset, load_metric

    metric_bleu = load_metric("../utils/bleu_metric.py")
    # metric_bleu.add_batch(predictions=[pred.split()], references=[[refs[0].split()]])
    # print(metric_bleu.compute(predictions=[pred], references=[refs]))
