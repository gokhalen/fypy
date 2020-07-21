import unittest
from ..libutil.util import *

class TestLibUtil(unittest.TestCase):
    
    def test_Point_distance(self):
        p1=Point([1,2,3])
        p2=Point([6,5,11])
        dd=p1.distance(p2)
        self.assertEqual(9.899494936611665,dd)
        
    
