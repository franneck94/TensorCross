"""Test vectorND code.
"""
import math
import unittest

from tensorcross import test_fn


class VectorTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        result = test_fn()
        self.assertEqual("hello world", result)


if __name__ == '__main__':
    unittest.main()
