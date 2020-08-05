import sys,os
import numpy as np
import matplotlib.pyplot as plt

# >>>>>>>>>>>>> INPUT SECTION <<<<<<<<<<<<
# specify .visit master list file that contains all .okc file names for a given visit session
def usage(): 
    print(' -- OKC Processor -- ')
    print(' Maps the various OKC files extracted from VisIt to a single files per variable with the format: ')
    print(' HEADER INFORMATION ')
    print('   i j k value ')
    print('   . . . ...')
    print('   . . . ...')
    print(' -- Usage -- ') 
    print(' python okcProc.py <visit file>')
    print(' Where <visit file> is the *.visit file that is dumped (by VisIt) when writing the OKC files.')
    print(' The <visit file> contains the list of all *.okc files dumped from the extraction.')

args = sys.argv

if len(args) == 1: 
    usage()
    sys.exit()


#input_visit_file = 'visit_ex_db.visit'
#input_visit_file = 'handoff_file.visit'
input_visit_file = args[1]
print(args)

print('Going to load file: ',input_visit_file)

# >>>>>>>>>>>>>  END  <<<<<<<<<<<<<<<<<<<<

print( " \n " )
print( " MASTER INPUT FILE: " + np.str(input_visit_file) )

# open the input file
input_file = open(input_visit_file, 'r')

# processing only the files which are included in this master list
file_list = input_file.readlines()[1:]

# close master list file
input_file.close()

# initialize list that will hold data from all *.okc files
comb_values = []


for i, item in enumerate(file_list):
	
   # open the first file in the series from the master list
   item = item.strip('\n')
   print( " \n " )
   print( " Processing file: " + item)
   print( " --------------------------------------- " )
   proc_file = open(item, 'r' )
   
   # read the header line, which contains only 3 integers: number of scalars in the file (including x, y, z),
   # number of lines of data, and number of something - not sure what the last value is
   num_var, num_lines, num_something = [int(x) for x in next(proc_file).split()]
   
   # read the subsequent n lines which contain the scalar names included in this file
   var_list = [next(proc_file).strip('\n') for x in range(num_var)]
   
   # make sure that each file contains (x,y,z) triplet as the first set of scalars
   if ( var_list[0] != 'x' ) or ( var_list[1] != 'y' ) or ( var_list[2] != 'z' ):
    print( ' \n ' )
    print( ' >>>>>>>>>>>>>>>>>>>> EXCEPTION <<<<<<<<<<<<<<<<<< ' )
    print( ' x, y, or z coordinate not found in the input file ' )
    print( ' ' + np.str(item) ) 
    print( ' Exiting program ................................. ' )
    print( ' >>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<< ' )
    exit()
   
   # skip n lines with the min/max values for each of the scalars above
   skip_minMaxVal_lines = [next(proc_file).split() for x in range(num_var)]

   # putin statements for czeching
   print(num_var, num_lines, num_something)
   print(var_list)
   # print(skip_minMaxVal_lines)
   
   # read data into list
   values = [[float(x) for x in line.split()] for line in proc_file]
   
   # keep appending lists to lists from all *.okc files
   if i == 0:
	   comb_values = values
   else:
	   comb_values.extend(values)

   # close *.okc file
   proc_file.close()


# Determine length of original list   
len_orig = len(comb_values)

# checking for duplicates and removing them
print( " \n " )
print( " Finished processing files \n" )
print( " Removing duplicates ..." )
nonDuplicate_values = [i for n, i in enumerate(comb_values) if i not in comb_values[:n]]

# Determine length of list without duplicates
len_ND = len(comb_values)

# Calculate number of duplicates
ND = len_orig - len_ND

print(' Number of duplicates: '+np.str(ND) )
print( " DONE " )

# convert to an array - with possible duplicates
print( " \n " )
print( " Data combined from all input files: " )
print( " ----------------------------------- " )
array = np.array(comb_values)
print(array.shape)

print( " \n " )
print( " Data combined from all input files with duplicate entries removed: " )
print( " ------------------------------------------------------------------ " )
# convert to a non-duplicate array
arrayND = np.array(nonDuplicate_values)
print(arrayND.shape)

########################################################################### 
# This portion of the code is only for flare handoff with 		 
# handoff being in the z-plane, and active x,y coordinates.
# It would be nice to write this section to find out which of the
# x,y,z coordinates from visit is zero, and then find out orientation
# of the plane and adjust (or map) the other two coordinates accordingly,
# i.e. find out mapping between x,y,z from the handoff simulation to i,j,k
# required for the receiving simulation				 
###########################################################################

# Plotting x,y coordinates to make sure the spacing is constant - visual inspection required
# change the axis plotting based on the dummy dimension; in this particular case z=0.0, so plotting x,y
# This was a quick addition ... maybe add all as subplots into one plot
plt.ion()
for i, item in enumerate(var_list[3:]):
        plt.figure()
        plt.scatter(arrayND[:,0],arrayND[:,1],c=arrayND[:,3+i],marker='s',cmap='hsv')
        plt.title(np.str(var_list[3+i]))
        plt.colorbar()
        plt.savefig(np.str(var_list[3+i]).split('/')[0]+'.pdf')
        # plt.waitforbuttonpress(timeout=1)
        
plt.ioff()
        
print( ' \n ' )
print( ' Calculating grid spacings: ' )
print( ' -------------------------- ' )

# Sort by z-axis, then y-axis, then x-axis
# This order may be modified based on the orientation of the handoff plane
# In this particular case, z is up and the handoff plane lies in the z-plane,
# and therefore the z-coordinate is 0.

# Sort z coordinate first and determine dz
a=arrayND[:,2].argsort(axis=0,kind='stable')
arrayND[:,:]=arrayND[a,:]
dz = 0.0
for i in range(arrayND.shape[0]-1):
        dz = arrayND[i+1,2]-arrayND[i,2]
        if abs(dz) > 1.e-10:
                  break
print('dz='+np.str(dz))

# Sort y coordinate and determine dy
a=arrayND[:,1].argsort(axis=0,kind='stable')
arrayND[:,:]=arrayND[a,:]	
dy = 0.0
for i in range(arrayND.shape[0]-1):
        dy = arrayND[i+1,1]-arrayND[i,1]
        if abs(dy) > 1.e-10:
                  break
# 	print(i,dy,arrayND[i,1])
print('dy='+np.str(dy))

# Sort x coordinate last and determine dx
a=arrayND[:,0].argsort(axis=0,kind='stable')
arrayND[:,:]=arrayND[a,:]
dx = 0.0
for i in range(arrayND.shape[0]-1):
        dx = arrayND[i+1,0]-arrayND[i,0]
        if abs(dx) > 1.e-10:
                  break
print('dx='+np.str(dx))
print('\n')
print(' Writing handoff files for all scalars: ')
print(' -------------------------------------- ')	
# Create handoff files for all scalars in the file
print(var_list)
for i, item in enumerate(var_list[3:]):

        item = item.strip('\n')
        item = item.strip('/0')
        file_handoff = "handoff_"+input_visit_file.rsplit(".",1)[0]+"_"+item+".dat"
        print('Creating file: '+file_handoff)
        
        fh=open(file_handoff,'w')
        fh.write(item+'\n')
        fh.write(np.str(dx)+' '+np.str(dy)+'\n')
        fh.write(np.str(arrayND.shape[0])+'\n')
        
        for j in range(arrayND.shape[0]):
                
                calc_i=int(round((arrayND[j,0]+dx/2)/dx))-1
                calc_j=int(round((arrayND[j,1]+dy/2)/dy))-1
                #calc_k=0

                fh.write(np.str(calc_i)+' '+np.str(calc_j)+' '+np.str(arrayND[j,i+3])+'\n')
                
        fh.close()
        
print('\n')
print(' ... DONE ... ')
print('\n')
