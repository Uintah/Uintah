# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:55:59 2013

@author: jeremy


"""

from mixing_table import ClassicTable as ct
import numpy as np

file_in = input('Enter filename: ')
myTable = ct(file_in)

name = myTable.ind_names[0]
num_dim = np.size(myTable.ind_names)

print(':::::Min/Max:::::')
print('(State-space is not guaranteed outside these bounds)')
print('  ' +np.str(myTable.ind_names[0])+':('+np.str(np.min(myTable.ind[0]))+','+np.str(np.max(myTable.ind[0]))+')')
if ( num_dim > 1):
    print('  '+np.str(myTable.ind_names[1])+':('+np.str(np.min(myTable.ind[1]))+','+np.str(np.max(myTable.ind[1]))+')')
    if ( num_dim == 3):
        print('  '+np.str(myTable.ind_names[2])+':('+np.str(np.min(myTable.ind[2]))+','+np.str(np.max(myTable.ind[2]))+')')
print('')        
        

 
doit = True
num_loops = 0

 
while doit == True:    

    if ( num_dim == 1 ):
        i1=input('Enter '+ myTable.ind_names[0]+':')
        x0 = np.array([np.float(i1)])
    elif ( num_dim == 2 ):
        i1=input('Enter '+ myTable.ind_names[0]+':')
        i2=input('Enter '+ myTable.ind_names[1]+':')
        x0 = np.array([np.float(i1),np.float(i2)])
    else:
        i1=input('Enter '+ myTable.ind_names[0]+':')
        i2=input('Enter '+ myTable.ind_names[1]+':')
        i3=input('Enter '+ myTable.ind_names[2]+':')
        x0 = np.array([np.float(i1),np.float(i2),np.float(i3)])
        
    counter = 0
    
    print('State space: ')
    state_space = myTable.interpolate(x0)
    for i, name in enumerate(myTable.dep_names):
      print(name, ' = ', state_space[i])
    
    print('\n')
    goon = input('Enter 0 to stop or any number to continue: ')
    print('\n')
    
    if ( np.float(goon) < 0.1 ):
        break
    
    
  

