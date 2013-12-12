# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:45:51 2013

@author: jeremy
"""

from mixing_table import ClassicTable as ct
import numpy as np
import matplotlib.pyplot as plt

file_in = raw_input('Enter filename: ')
myTable = ct(file_in)

name = myTable.ind_names[0]
num_dim = np.size(myTable.ind_names)

num_dim = np.size(myTable.ind_names)

print(':::::Min/Max:::::')
print('(State-space is not guaranteed outside these bounds)')
print('  ' +np.str(myTable.ind_names[0])+':('+np.str(np.min(myTable.ind[0]))+','+np.str(np.max(myTable.ind[0]))+')')
if ( num_dim > 1):
    print('  '+np.str(myTable.ind_names[1])+':('+np.str(np.min(myTable.ind[1]))+','+np.str(np.max(myTable.ind[1]))+')')
    if ( num_dim == 3):
        print('  '+np.str(myTable.ind_names[2])+':('+np.str(np.min(myTable.ind[2]))+','+np.str(np.max(myTable.ind[2]))+')')
print('')    

print('Choose a plot:')
print('1. (x)'+myTable.ind_names[0]+' vs. (y) dep. variable with changing: '+myTable.ind_names[1])
print('2. (x)'+myTable.ind_names[0]+' vs. (y) dep. variable with changing: '+myTable.ind_names[2])
print('3. (x)'+myTable.ind_names[1]+' vs. (y) dep. variable with changing: '+myTable.ind_names[2])
option = raw_input('Option: ')

if ( np.int(option) == 1):
    ng1 = raw_input('Num points in:'+myTable.ind_names[0]+': ')
    ng2 = raw_input('Num points in:'+myTable.ind_names[1]+': ')
    fixval_i = raw_input('Enter a constant '+myTable.ind_names[2]+':')
    i1 = 0
    i2 = 1
    ifix = 2

elif ( np.int(option) == 2): 
    ng1 = raw_input('Num points in:'+myTable.ind_names[0]+': ')
    ng2 = raw_input('Num points in:'+myTable.ind_names[2]+': ')
    fixval_i = raw_input('Enter a constant '+myTable.ind_names[1]+':')
    i1 = 0
    i2 = 2
    ifix = 1
else: 
    ng1 = raw_input('Num points in '+myTable.ind_names[1]+': ')
    ng2 = raw_input('Num points in '+myTable.ind_names[2]+': ')
    fixval_i = raw_input('Enter a constant '+myTable.ind_names[0]+': ')
    i1 = 1
    i2 = 2
    ifix = 0
    
fixval = np.float(fixval_i)    

counter = 0        
for name in myTable.dep_names:
    
    print(np.str(counter)+ ' = '+ np.str(myTable.dep_names[counter]))
    counter += 1    

iD_i = raw_input('Enter a dep. var index: ')
iD = np.int(iD_i)    
    
i1_min = np.min(myTable.ind[i1])
i1_max = np.max(myTable.ind[i1])
i2_min = np.min(myTable.ind[i2])
i2_max = np.max(myTable.ind[i2])

dx1 = (i1_max-i1_min)/np.float(ng1)
dx2 = (i2_max-i2_min)/np.float(ng2)

i2V = i2_min
color=0
dcolor = 1.0/np.float(ng2)
for i in range (0,np.int(ng2)+1):
    plot_me = []
    plot_me_x=[]
    i1V = i1_min
    for j in range (0,np.int(ng1)+1):
        
        x=[0,0,0]
        x[i1]=i1V
        x[i2]=i2V
        x[ifix]=fixval
        
        plot_me.append(myTable.interpolate(x,iD))
        plot_me_x.append(i1V)
        
        i1V += dx1
        
            
    plt.plot(plot_me_x,plot_me,'--',c='.6')
    plt.scatter(plot_me_x,plot_me,c=np.str(color),s=40)

    color += dcolor
    i2V += dx2   

plt.xlabel(myTable.ind_names[i1])
plt.ylabel(myTable.dep_names[iD]) 
plt.title(myTable.dep_names[iD]+'=f('+myTable.ind_names[i1]+ ','+myTable.ind_names[i2]+')')       
plt.grid()   
print('')
print('Colors go (low->high), (dark->light) for '+myTable.ind_names[i2]+')') 
plt.show()



        
        
        

    
 
