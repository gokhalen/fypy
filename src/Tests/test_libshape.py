import unittest
from ..libshape.shape import *


class TestLibShape(unittest.TestCase):
    
    def test_shape1d(self):
        print('Executing test_shape1d...')
        self.assertEqual((0,1),shape1d(1))
        self.assertEqual((1,0),shape1d(-1))
        self.assertEqual((0.5,0.5),shape1d(0))
        print('Executed test_shape1d...')

    def test_shape1d_der(self):
        print('Executing test_shape1d_der...')
        self.assertEqual((-0.5,0.5),shape1d_der(-1))
        self.assertEqual((-0.5,0.5),shape1d_der(1))
        self.assertEqual((-0.5,0.5),shape1d_der(1))
        print('Executed test_shape1d_der...')


