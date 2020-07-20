import unittest
from ..libshape.shape import *


class TestLibShape(unittest.TestCase):
    
    def test_shape1d(self):
        self.assertEqual((0,1),shape1d(1))
        self.assertEqual((1,0),shape1d(-1))
        self.assertEqual((0.5,0.5),shape1d(0))

    def test_shape1d_der(self):
        self.assertEqual((-0.5,0.5),shape1d_der(-1))
        self.assertEqual((-0.5,0.5),shape1d_der(1))
        self.assertEqual((-0.5,0.5),shape1d_der(1))


