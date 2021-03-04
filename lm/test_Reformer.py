from unittest import TestCase

import pytest
from Reformer import Reformer


@pytest.fixture
def reformer():
    return Reformer()

class TestReformer(TestCase):

    def test_pred(self, reformer):
        encoded, attention_masks = reformer.encode(["this is a"])
        reformer.pred(encoded)

    def test_generate(self, model):
        reformer.generate("test")

"""
import importlib
import Reformer
importlib.reload(Reformer)
from Reformer import Reformer
reformer = Reformer()
encoded, attention_masks = reformer.encode(["this is a"])
r = reformer.pred(encoded)

encoded, attention_masks = reformer.encode([text])
r = reformer.pred(encoded)

"""
