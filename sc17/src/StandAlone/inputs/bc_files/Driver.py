# -*- coding: utf-8 -*-
# import modules
import os
from bc_functions import *

###-----------------Required User inputs-----------------------###
original_ups_file='stan_phil_derek.ups'
ups_file='prediction.ups'  # name of new input file that will be created
coal_filename='btc.xml'
file_directory='bc_files/'
mass_flow_factor=1.0000000
###-----------------Optional User inputs-----------------------###
LuseTable=0
if LuseTable :
    table_name = 'btc.mix'
else :
    R=8314.46                       # J / kmol K
    MW=34.4                         # kg / kmol
    T_Fahrenheit= 500               # F
    T=(T_Fahrenheit-32)/9*5+273.15  # K
    P=101325                        # J / m^3
    density =P/R/T*MW               # kg / m^3
    print "Using Density", density, "kg / m^3"
###------------------------------------------------------------###


print ""
print ""


### CREATE BC (sub-ups) FILES
# copy original ups file to new file:
os.system("cp " + original_ups_file + " " + ups_file)
# read in coal input file, create coal object, and generate coal input file:
myCoal = Coal(coal_filename)
table_base_name='emaxxxple_input_file.dat'
myCoal.create_table_input_file(table_base_name)
# read in ups file and get faces
myFace = Face(ups_file)
# create mixing table object
if LuseTable :
    mixing_table = ct(table_name)
 


# loop over all faces in input file
count = 0
for x in myFace.input_files:
    #x=myFace.input_files[8]
    print ""
    print ""
    print "Creating input file for Face ", count, " of ", np.size(myFace.input_files) - 1, ": ", x
    # load inputs from bc input file
    myBC = BC(x, myFace.names[count])
    # Modity mass flow rates if coal moisture is present
    myBC.mdot_modify(myCoal,mass_flow_factor )
    # compute bc velocity
    if LuseTable :
        gas_vel = myFace.gas_velocity(myBC, mixing_table, count) # m/s
    else :
        gas_vel = myFace.gas_velocity_no_lookup(myBC,  count, density)
    # compute gas phase BCs
    gas = myFace.gas_phase_BCs(myBC, gas_vel, count)
    # compute dqmom BCs
    DQMOM = myFace.DQMOM_BCs(myCoal, myBC, ups_file, gas_vel, count)
    # write bc file
    myFace.write_bc_file(gas, DQMOM, count,file_directory)
    count += 1
print ""
print ""    
# re-write ups file
myFace.rewrite_ups(ups_file,file_directory)
print myCoal.dh_daf
print ""
print ""
print "Finished"


