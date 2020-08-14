import sys
sys.path.append("/Users/jeremy/Code/Uintah/src/StandAlone/inputs/ARCHES/handoff/scripts")
sys.path.append("/Users/jeremy/Code/Uintah/src/StandAlone/inputs/ARCHES/ClassicMixingTables/scripts")
from mixing_table import ClassicTable as ct
import ArchesHandoffGenerator as A 
import matplotlib.pyplot as plt
import numpy as np

def get_mdot( Nspace, fp, eta, hl, v, myTable, area, R ): 

    """ Compute the mdots if v > 0. """

    mdot = 0
    mdot_eta = 0
    mdot_fp = 0
    mdot_fuel = 0
    density = np.zeros((Nspace[0],Nspace[1]))

    for i in range(Nspace[0]): 
        for j in range(Nspace[1]): 
    
            f = fp[i,j]/(1.-eta[i,j])
            x = [eta[i,j],hl[i,j],f]
    
            state_space = myTable.interpolate(x)

            density[i,j] = state_space[2]
    
            if v[i,j] > 0: 
                mdot      += state_space[2]*v[i,j]*(1+R)*area
                mdot_fp   += fp[i,j]*state_space[2]*v[i,j]*(1+R)*area
                mdot_eta  += eta[i,j]*state_space[2]*v[i,j]*(1+R)*area
                mdot_fuel += fp[i,j]*state_space[2]*v[i,j]*(1+R)*area \
                           + eta[i,j]*state_space[2]*v[i,j]*(1+R)*area

    return mdot, mdot_eta, mdot_fp, mdot_fuel, density

#---------------------------------------

orig_data_i = 1
int_i = 0

filenames = ['handoff_fine_heat_loss.dat', 'handoff_fine_uVelocitySPBC.dat',  
             'handoff_fine_mixture_fraction_2.dat','handoff_fine_mixture_fraction.dat','handoff_fine_enthalpy.dat']

ho_info = {}
for f in filenames: 
    this_info = []
    this_name = (f.split('.dat')[0]).split('handoff_fine_')[1].split('.')[0]
    interpolator,x,y,newX,newY,origdata = A.getInterpolator(f, 2)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dx_new = newX[1]-newX[0]
    dy_new = newY[1]-newY[0]
    this_info.append(interpolator)
    this_info.append(origdata)
    ho_info[this_name] = this_info

area_cell = dx*dy
area_new_cell = dx_new*dy_new

#lets compute the total flow rates of the various variables from the original data: 
N = np.shape(ho_info['mixture_fraction'][orig_data_i])
Nnew = [np.shape(newX)[0], np.shape(newY)[0]]

#load the table: 
myTable = ct('CH4_gogolek.mix')

#compute the mdots
fp = ho_info['mixture_fraction_2'][orig_data_i]
eta = ho_info['mixture_fraction'][orig_data_i]
hl = ho_info['heat_loss'][orig_data_i]
v = ho_info['uVelocitySPBC'][orig_data_i]

mdot, mdot_eta, mdot_fp, mdot_fuel, rho_orig = get_mdot(N,fp,eta,hl,v,myTable,area_cell, 0.0)

#compute the mdot on the interpolated mesh: 
fp_i = ho_info['mixture_fraction_2'][int_i](newX,newY)
eta_i = ho_info['mixture_fraction'][int_i](newX,newY)
heat_loss_i = ho_info['heat_loss'][int_i](newX,newY)
v_i = ho_info['uVelocitySPBC'][int_i](newX,newY)

mdot_new, mdot_eta_new, mdot_fp_new, mdot_fuel_new, rho_coarse = get_mdot(Nnew,fp_i,eta_i,heat_loss_i,v_i,myTable,area_new_cell, 0.)

#compute the ratios for adjustment: 
r_mdot = ( mdot - mdot_new ) / mdot
r_mdot_fp = ( mdot_fp - mdot_fp_new ) / mdot_fp
r_mdot_fuel = ( mdot_fuel - mdot_fuel_new ) / mdot_fuel

#which R?
R = r_mdot_fp

print('Adjustment ratio, R = {}'.format(R))
print('mdot_fp = ', mdot_fp)
print('mdot_fp_new =', mdot_fp_new)

mdot_new, mdot_eta_new, mdot_fp_new, mdot_fuel_new, rho_coarse = get_mdot(Nnew,fp_i,eta_i,heat_loss_i,v_i,myTable,area_new_cell, R)

print('Mdot error: ', (mdot-mdot_new)/mdot*100.,'%')
print('Mdot fp error: ', (mdot_fp-mdot_fp_new)/mdot_fp*100.,'%')
print('Mdot eta error: ', (mdot_eta-mdot_eta_new)/mdot_eta*100.,'%')
print('Mdot total fuel error: ', (mdot_fuel-mdot_fuel_new)/mdot_fuel*100.,'%')

#write out the interpolated values: 
f = open('w.handoff','w')
f.write('CCWVelocity\n')
f.write('{} {}\n'.format(dx_new, dy_new))
f.write('{}\n'.format(Nnew[0]*Nnew[1]))

for i in range(Nnew[0]): 
    for j in range(Nnew[1]): 

        source = rho_coarse[i,j]*v_i[i,j]*v_i[i,j]

        f.write('0 {} {} {}\n'.format(i,j,source))
        #if source > 0: 
            #f.write('0 {} {} {}\n'.format(i,j,source))
        #else: 
            #f.write('0 {} {} {}\n'.format(i,j,0.))


f.close()

f = open('mixture_fraction_fp.handoff','w')
f.write('mixture_fraction_fp\n')
f.write('{} {}\n'.format(dx_new, dy_new))
f.write('{}\n'.format(Nnew[0]*Nnew[1]))

for i in range(Nnew[0]): 
    for j in range(Nnew[1]): 

        source = rho_coarse[i,j]*v_i[i,j]*fp_i[i,j]

        if fp_i[i,j] > 0: 
            f.write('0 {} {} {}\n'.format(i,j,source))
        else: 
            f.write('0 {} {} {}\n'.format(i,j,0.))

f.close()

f = open('mixture_fraction.handoff','w')
f.write('mixture_fraction\n')
f.write('{} {}\n'.format(dx_new, dy_new))
f.write('{}\n'.format(Nnew[0]*Nnew[1]))

for i in range(Nnew[0]): 
    for j in range(Nnew[1]): 

        source = rho_coarse[i,j]*v_i[i,j]*eta_i[i,j]

        if eta_i[i,j] > 0: 
            f.write('0 {} {} {}\n'.format(i,j,source))
        else: 
            f.write('0 {} {} {}\n'.format(i,j,0.))

f.close()

f = open('heat_loss.handoff','w')
f.write('heat_loss\n')
f.write('{} {}\n'.format(dx_new, dy_new))
f.write('{}\n'.format(Nnew[0]*Nnew[1]))

for i in range(Nnew[0]): 
    for j in range(Nnew[1]): 

        f.write('0 {} {} {}\n'.format(i,j,heat_loss_i[i,j]))

f.close()

#need to make one for the enthalpy source: 
f = open('enthalpy.handoff','w')
f.write('enthalpy\n')
f.write('{} {}\n'.format(dx_new, dy_new))
f.write('{}\n'.format(Nnew[0]*Nnew[1]))
for i in range(Nnew[0]): 
    for j in range(Nnew[1]): 

        m = fp[i,j]/(1.-eta[i,j])
        x = [eta[i,j],hl[i,j],m]
        
        state_space = myTable.interpolate(x)

        h_sens = state_space[4]
        h_ad = state_space[5]

        h = h_ad - heat_loss_i[i,j]*h_sens

        source = rho_coarse[i,j]*v_i[i,j]*h

        f.write('0 {} {} {}\n'.format(i,j,source))
        #if source > 0: 
            #f.write('{} {} 0 {}\n'.format(i,j,source))
        #else: 
            #f.write('{} {} 0 {}\n'.format(i,j,0.))

f.close()

print('(0,0) location: {},{}'.format(newX[0],newY[0]))

#check = fint(x,y)

#plt.contourf(x,y,check)
#plt.contourf(x,y,orig_Data)

#check2 = fint(newX,newY)
#plt.contour(newX,newY,check2,colors='k')

#-------------

#plt.show()
