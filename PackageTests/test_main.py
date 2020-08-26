'''
Nachiket Gokhale gokhalen@gmail.com
For this to work, the tests must be discovered from the fypy root directory
Test directories must be packages i.e. should have __init__.py file. Test files must be named starting from with "test_" and test classes must inherit from unittest.TestCase. Functions inside the test class which begin with test_ are executed automatically. 
'''
import os,sys
import unittest

TestDir='PackageTests'

class TestMain(unittest.TestCase):
    def test_main(self):
        #os.system(f'python3.8 main.py "" {TestDir}')
        #with open(f'{TestDir}'+'/'+'data.out',mode='r') as f:
        #    line=f.readline()
        #self.assertEqual(line,'Dummy data')
        pass
    
