#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

if __name__ == "__main__":
  #Changes the entime in uda_path/input.xml to the new end time
  #also set the artificial damping coefficient in ALL checkpoints
  #to zero
  #Argumented as:	 setup_restart.py uda_dir 
  #
  #end time must be manually changed below. 
  # FYI: NO BULLETPROOFING USE AT OWN RISK1!!!
  new_end_time = '900e-6'
  
  #get uda directory
  uda_dir = os.path.abspath(sys.argv[1])
  
  #Fix input file
  input_file = uda_dir+'/input.xml'
  F = open(input_file,"r+")
  all_lines = F.read().split('\n')
  F.seek(0)
  for line in all_lines:
    if '<maxTime>' in line:
      line = '    <maxTime>'+new_end_time+'</maxTime>'
    F.write(line+'\n')
  F.close()
  
  #Find checkpoint dirs and change damping
  for item in os.listdir(uda_dir+"/checkpoints/"):
    tmp_file = uda_dir+"/checkpoints/"+item+"/timestep.xml"
    if os.path.isfile(tmp_file):
      F = open(tmp_file,"r+")
      all_lines = F.read().split('\n')
      F.seek(0)
      for line in all_lines:
	if '<artificial_damping_coeff>' in line:
	  line = '    <artificial_damping_coeff>0.0</artificial_damping_coeff>'
	F.write(line+'\n')
      F.close()
	
      