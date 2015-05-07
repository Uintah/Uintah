# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 13:45:30 2014

@author: tsaad
"""
import os
def uintahcleanup():
  cleanup = 'rm -rf *.uda*'
  os.system(cleanup)
  cleanup = 'rm -rf *.dot'
  os.system(cleanup)
  cleanup = 'rm -rf *-t*.ups'
  os.system(cleanup)
  cleanup = 'rm -rf *-t*.txt'
  os.system(cleanup)