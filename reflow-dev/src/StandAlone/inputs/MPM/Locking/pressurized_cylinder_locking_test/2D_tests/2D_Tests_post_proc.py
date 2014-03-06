#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

#Subprocess module, calls external executables
import subprocess as sub_proc  

#Plotting stuff
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import hsv_to_rgb

#Set matplotlib defaults to desired values
#Set the legend to best fit
fontSize = 24
markers = None
plt.rcParams['legend.loc']='best'
#Set font size
plt.rcParams['font.size']=fontSize
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelsize']='large'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['legend.fontsize']='medium'
#Set linewidth
plt.rcParams['lines.linewidth']=4
#Set markersize
plt.rcParams['lines.markersize'] = 8
#Set padding for tick labels and size
plt.rcParams['xtick.major.pad']  = 12
plt.rcParams['ytick.major.pad']  = 8
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 4

#resolution
plt.rcParams['figure.dpi']=120

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : fontSize}
rc('font', **font)

#X and Y grid divisions (Major)
# for minor multiply by 4
yDiv = 10
xDiv = 10


# ---------- User input parameters ---------- #
a = 0.20				#Inside radius (m)
b = 0.50				#Outside radius (m)
Pb = 6.89476e6				#Pressure applied at outside radius (Pa)
Pa = 0.0				#Pressure applied at inside radius (Pa)
K = 70.28e9				#Bulk modulus of the material (Pa)
G = 26.23e9				#Shear modulus of the material (Pa)
numSteps = 900				#Number of simulation steps
cellSize = 0.005				#Cell Spacing (m)

# ---------- --------------------- ---------- #
#Calculate other material properties
nu = (3.0*K-2.0*G)/(2.0*(3.0*K+G))	#Poisons ratio
E = (9.0*K*G)/(3.0*K+G)			#Youngs modulus

def savePNG(name,size='1920x1080',res=80):
  #Add Check for file already existing as name.png
  if size == '640x480':
    size = [8,6]
    res = 80
  if size == '1080x768':
    size = [13.5,9.6]
    res = 80
  if size == '1280x960':
    size = [16,12]
    res = 80
  if size == '1920x1080':
    size = [24,13.5]
    res = 80    
  #set the figure size for saving
  plt.gcf().set_size_inches(size[0],size[1])
  #save at speciified resolution
  plt.savefig(name+'.png', bbox_inches=0,dpi=res) 

def print_properties():
    global Pb,a,b,E,nu
    print '#'*40,'\n'
    print 'Cylinder inside radius: ',a,' (m)'
    print 'Cylinder outer radius:  ',b,' (m)'
    print ''
    print 'Cylinder inside pressure: ',format(Pa,'1.4e'),' (Pa)'
    print 'Cylinder outer pressure:  ',format(Pb,'1.4e'),' (Pa)'
    print ''
    print 'Bulk Modulus:   ',format(K,'1.4e'),' (Pa)'
    print 'Shear Modulus:  ',format(G,'1.4e'),' (Pa)'
    print 'Youngs Modulus: ',format(E,'1.4e'),' (Pa)'
    print 'Poissons Ratio: ',format(nu,'1.4f'),' (unitless)'
    print '\n','#'*40,'\n'

def mod_properties(nu):
  base_K = 70.28e9
  base_G = 26.23e9
  base_nu = 0.3756
  
  new_K = ((2.0/3.0)*base_G*(1.0+nu))/(1.0-(2.0*nu))
  new_G = ((3.0/2.0)*base_K*(1.0-(2.0*nu)))/(1.0+nu)
  
  print '#'*40,'\n'
  print 'Base Properties:'
  print '\tBulk Modulus:   ',format(base_K,'1.4e'),' (Pa)'
  print '\tShear Modulus:  ',format(base_G,'1.4e'),' (Pa)'
  print '\tPoissons Ratio: ',base_nu,' (unitless)'
  print '\nTarget nu:',format(nu,'1.4f'),' (unitless)'
  print '\nUsing Modified Bulk Modulus:'
  print '\tBulk Modulus:   ',format(new_K,'1.4e'),' (Pa)'
  print '\nUsing Modified Shear Modulus:'
  print '\tShear Modulus:  ',format(new_G,'1.4e'),' (Pa)'
  print '\n','#'*40

def epsil_rr(r):
  global Pa,Pb,a,b,E,nu
  a_sqr = pow(a,2.0)
  b_sqr = pow(b,2.0)
  r_sqr = pow(r,2.0)
  epsil_r = ((1.0+nu)*a_sqr*b_sqr)/(E*(b_sqr-a_sqr))*(((Pb+Pa)/r_sqr)+(1.0-2.0*nu)*((Pa*a_sqr-Pb*b_sqr)/(a_sqr*b_sqr)))
  return epsil_r
  
def epsil_thetatheta(r):
  global Pa,Pb,a,b,E,nu
  a_sqr = pow(a,2.0)
  b_sqr = pow(b,2.0)
  r_sqr = pow(r,2.0)
  epsil_theta = ((1.0+nu)*a_sqr*b_sqr)/(E*(b_sqr-a_sqr))*(((Pa-Pb)/r_sqr)+(1.0-2.0*nu)*((Pa*a_sqr-Pb*b_sqr)/(a_sqr*b_sqr)))
  return epsil_theta

if __name__ == "__main__":
  #execute as follows
  #		script			uda folder	       spacing  K   G
  #  python ./path_to/max_disp.py ./path_to/uda_folder.uda.000/ 0.01 70e9 26e6 
  
  
  #uda path from command line argumentation
  uda_path = os.path.abspath(sys.argv[1])
  #Subdir
  #save_path = os.path.split(uda_path)[0]+'/image_set_2'
  save_path = os.environ['TESTS_2D_IMG_DIR']
  #Image name
  img_name = './displacement_comparison'
  
  
  #Get cell size
  cellSize = float(sys.argv[2])
  #Get Bulk modulus
  K = float(sys.argv[3])
  #Get Shear Modulus
  G = float(sys.argv[4])
  #Recompute poisson ratio and youngs modulus
  nu = (3.0*K-2.0*G)/(2.0*(3.0*K+G))		#Poisons ratio
  E = (9.0*K*G)/(3.0*K+G)			#Youngs modulus
  
  if K == 70.28e9 and G == 26.23e9:
    prepend = 'Aluminum_'
  elif K == 70.28e9 and G != 26.23e9:
    prepend = 'ModifiedShearModulus_'
  elif K != 70.28e9 and G == 26.23e9:
    prepend = 'ModifiedBulkModulus_'
  else:
    prepend = 'ERROR_'

  
  img_name = './'+prepend+'displacementComparison_Spacing_'+format(cellSize,'1.4f')+'_poisson_'+format(nu,'1.4f')

  #Dont Extract if specified
  if len(sys.argv) == 6:
    #Dont extract just read exisiting files
    F_initial = open("./initial_position","r")
    F_final = open("./final_position","r")    
  else:
    #Do extraction
    F_initial = open("./initial_position","w+")
    F_final = open("./final_position","w+")  
    
    #Extract the initial and final particle positions
    print '\nExtracting initial particle positions...'
    args_initial = ["partextract","-partvar","p.x","-timesteplow","0","-timestephigh","0",uda_path]
    tmp = sub_proc.Popen(args_initial,stdout=F_initial,stderr=sub_proc.PIPE)
    dummy = tmp.wait()
    F_initial.seek(0)
    print('Done.')

    print('\nExtracting final particle positions...')
    args_final =   ["partextract","-partvar","p.x","-timesteplow",str(numSteps),"-timestephigh",str(numSteps),uda_path]
    tmp = sub_proc.Popen(args_final,stdout=F_final,stderr=sub_proc.PIPE)
    dummy = tmp.wait()
    F_final.seek(0)
    print('Done.')

  #Make/and or move into folder for images
  try:
    os.mkdir(save_path)
    print '\nSpawned folder successfully.'
  except:
    print '\nError: Failed to make image directory.\n\tPATH=',save_path,'\n'
  try:
    os.chdir(save_path)
    print '\nEntered image folder successfully.'
  except:
    print '\nError: Failed to move into image directory.\n\tPATH=',save_path,'\n'

  #Build a dictionary of the initial particle positions.
  initial_dict = {}
  for line in F_initial:
    line = line.strip().split()
    pID = int(line[3])
    pX = float(line[4])
    pY = float(line[5])
    pZ = float(line[6])
    initial_dict[pID] = [pX,pY,pZ]
  F_initial.close()

  #Build a dictionary of the final particle positions.
  final_dict = {}  
  for line in F_final:
    line = line.strip().split()
    pID = int(line[3])
    pX = float(line[4])
    pY = float(line[5])
    pZ = float(line[6])
    final_dict[pID] = [pX,pY,pZ]
  F_final.close()

  #Build dictionary containing the displacements
  diff_dict = {}
  pID_list = list(initial_dict.keys())
  pID_list.sort()
  for pID in pID_list:
    initial = initial_dict[pID]
    final = final_dict[pID]
    dx = final[0]-initial[0]
    dy = final[1]-initial[1]
    dz = final[2]-initial[2]
    diff_dict[pID] = [dx,dy,dz]
    #print [dx,dy,dz]

  #max_dx = 0
  #max_dy = 0
  #max_dz = 0
  #max_dict = {}
  #for pID in pID_list:
    #diff = diff_dict[pID]
    #if abs(diff[0])>max_dx:
      #max_dx = abs(diff[0])
      #max_dict['dx'] = {'pID':pID,'initial':initial_dict[pID],'final':final_dict[pID],'diff':diff_dict[pID]}
    #if abs(diff[1])>max_dy:
      #max_dy = abs(diff[1])
      #max_dict['dy'] = {'pID':pID,'initial':initial_dict[pID],'final':final_dict[pID],'diff':diff_dict[pID]}
    #if abs(diff[2])>max_dz:
      #max_dz = abs(diff[2])
      #max_dict['dz'] = {'pID':pID,'initial':initial_dict[pID],'final':final_dict[pID],'diff':diff_dict[pID]}
  #print ''
  #for key in max_dict['dx']:
    #print key,':',max_dict['dx'][key]


  MAX_ERR =-1e99
  MAX_PERCENT_ERR = -1e99

  MIN_ERR = 1e99
  MIN_PERCENT_ERR = 1e99

  #Build a dictionary of unique initial radial positions. 
  #Contains dictionary of particles each having initial final and diff
  position_dict = {}
  for pID in pID_list:
    initial = initial_dict[pID]
    final = final_dict[pID]
    
    simulation_disp = diff_dict[pID][0]
    analytical_disp = epsil_thetatheta(initial[0])*initial[0]
    error = simulation_disp-analytical_disp
    percent_error = (error/analytical_disp)*100

    MAX_ERR = max(MAX_ERR,error)
    MAX_PERCENT_ERR = max(MAX_PERCENT_ERR,percent_error)
    MIN_ERR = min(MIN_ERR,error)
    MIN_PERCENT_ERR = min(MIN_PERCENT_ERR,percent_error)

    if initial[0] not in position_dict:
      position_dict[initial[0]] = {'particles':{pID:{'initial':initial,'final':final,'simulation_disp':simulation_disp,'error':error,'percent_error':percent_error}}}
    else:
      position_dict[initial[0]]['particles'][pID] = {'initial':initial,'final':final,'simulation_disp':simulation_disp,'error':error,'percent_error':percent_error}
      
  #Go through the radial position dictionary and add a summary containing
  # analytical solution, mean error and error extends, also L2 error
  rs_list = list(position_dict.keys())
  rs_list.sort()
  position_dict['total_L2_Err'] = 0.0
  for r in rs_list:
    tmp_pID_list = list(position_dict[r]['particles'].keys())
    num_tmp_pIDs = len(tmp_pID_list)
    
    tmp_maxDisp = -1e99
    tmp_minDisp = 1e99
    tmp_maxErr = -1e99
    tmp_minErr = 1e99
    
    running_sum_Disp= 0
    running_sum_Err = 0
    running_sum_L2_Err = 0
    
    position_dict[r]['maxDisp'] = {'pID':None,'val':tmp_maxDisp}
    position_dict[r]['minDisp'] = {'pID':None,'val':tmp_minDisp}
    position_dict[r]['maxErr'] = {'pID':None,'val':tmp_maxErr}
    position_dict[r]['minErr'] = {'pID':None,'val':tmp_minErr}
    position_dict[r]['meanDisp'] = None  
    position_dict[r]['meanErr'] = None
    position_dict[r]['meanErr_percent'] = None
    position_dict[r]['L2_Err'] = None

    
    for pID in tmp_pID_list:
      pID_disp = position_dict[r]['particles'][pID]['simulation_disp']
      pID_err = position_dict[r]['particles'][pID]['error']
      
      tmp_maxDisp = max(tmp_maxDisp,pID_disp)
      tmp_minDisp = min(tmp_minDisp,pID_disp)
      tmp_maxErr = max(tmp_maxErr,pID_err)
      tmp_minErr = min(tmp_minErr,pID_err)
      
      running_sum_Disp += pID_disp
      running_sum_Err += pID_err
      running_sum_L2_Err += pID_err**2
      
      if tmp_maxDisp == pID_disp:
	position_dict[r]['maxDisp']['pID'] = pID
	position_dict[r]['maxDisp']['val'] = pID_disp
      if tmp_minDisp == pID_disp:
	position_dict[r]['minDisp']['pID'] = pID
	position_dict[r]['minDisp']['val'] = pID_disp      
      if tmp_maxErr == pID_err:
	position_dict[r]['maxErr']['pID'] = pID
	position_dict[r]['maxErr']['val'] = pID_err 
      if tmp_minErr == pID_err:
	position_dict[r]['minErr']['pID'] = pID
	position_dict[r]['minErr']['val'] = pID_err
	
    position_dict[r]['meanDisp'] = running_sum_Disp/float(num_tmp_pIDs)
    position_dict[r]['meanErr'] = running_sum_Err/float(num_tmp_pIDs)
    position_dict[r]['meanErr_percent'] = (position_dict[r]['meanErr']/(epsil_thetatheta(r)*r))*100.0
    position_dict[r]['L2_Err'] = np.sqrt(running_sum_L2_Err)
    position_dict['total_L2_Err'] += running_sum_L2_Err
    
  num_rs = len(rs_list)
  position_dict['total_L2_Err'] = np.sqrt(position_dict['total_L2_Err'])
    
  #Go through dictionary and strip out plot info to appropriate lists
  tot_L2_err = position_dict['total_L2_Err']	#total L2 error over whole domain
  analytical_disp = []				#Analytical solution corresponding to rs
  mean_disp = []					#mean simulation solution corresponding to rs
  disp_plusErr = []				# + error bar extents
  disp_minusErr = []				# - error bar extents
  mean_err = []					#mean error between simulation analytical solution corresponding to rs
  mean_err_percent = []
  err_plusErr = []				# + error bar extents
  err_minusErr = []				# - error bar extents
  L2s = []  					#L2 error at a given radius
  for r in rs_list:
    analytical_disp.append(epsil_thetatheta(r)*r)
    mean_disp.append(position_dict[r]['meanDisp'])
    disp_plusErr.append(position_dict[r]['maxDisp']['val']-position_dict[r]['meanDisp'])
    disp_minusErr.append(position_dict[r]['meanDisp']-position_dict[r]['minDisp']['val'])
    mean_err.append(position_dict[r]['meanErr'])
    mean_err_percent.append(position_dict[r]['meanErr_percent'])
    err_plusErr.append(position_dict[r]['maxErr']['val']-position_dict[r]['meanErr'])
    err_minusErr.append(position_dict[r]['meanErr']-position_dict[r]['minErr']['val'])
    L2s.append(position_dict[r]['L2_Err'])

  print_properties()

  labels = ("analytical displacement","simulation displacement","percent error")

  fig1 = plt.figure(1)
  plt.title('Elastic Cylinder Comparison:\n' r'$\mathbf{spacing='+str(cellSize)+'}$' ' , ' r'$\mathbf{ppc=(2,2,1)}$' '\n' r'$\mathbf{\nu = '+format(nu,'1.3f')+'}$' ' , ' r'$\mathbf{P_b = '+format(Pb,'1.3e')+'}$')
  #Add first axis using subplot
  ax1 = fig1.add_subplot(111)
  line1 = plt.plot(rs_list,analytical_disp,linestyle='-',color='k',marker=markers,label='analytical displacement')
  line2 = plt.plot(rs_list,mean_disp,linestyle='-',color='b',marker=markers,label='simulation displacement')
  
  
  #line2 = plt.errorbar(rs_list, mean_disp,
  #           yerr=[disp_minusErr,disp_plusErr],
  #           marker='.',
  #           color='b',
  #           ecolor='r',
  #           markerfacecolor='b',
  #           label="simulation displacement",
  #           capsize=5,
  #           linestyle='-')
  
  ax1.yaxis.tick_left()
  plt.ylabel("Radial displacement (m)")
  plt.xlabel("Radial position (m)")

  #Now do the second axes that shares the x-axis with ax1
  ax2 = fig1.add_subplot(111,sharex=ax1, frameon=False)
  line3 = plt.plot(rs_list,mean_err_percent,linestyle='-',color='r',marker=markers,label='error')
  ax2.yaxis.tick_right()
  ax2.yaxis.set_label_position("right")
  plt.ylabel("Error (%)")

  fig1.legend((line1,line2,line3),labels)
  
  tot_L2 = position_dict['total_L2_Err']
  max_analytical_Disp = max(analytical_disp)
  max_at_r = rs_list[analytical_disp.index(max_analytical_Disp)]
  
  corner_text_string = 'Total L2 Error : '+format(tot_L2,'1.3e')+'\nMax analytical displacement:\n'+format(max_analytical_Disp,'1.3e')+' @ r='+format(max_at_r,'1.3f')+'\nScaled L2 Error : '+format(tot_L2/abs(max_analytical_Disp),'1.3e')
  plt.figtext(0.15,0.98,corner_text_string,ha='left',va='top',size='x-small')

  plt.draw()
  savePNG(img_name,size='1920x1080')
  #Convert to smaller image
  args_convert =   ["convert","1080x768",img_name+'.png',"-resize","1080x768",'./small/'+img_name+'_SMALL.png']
  tmp = sub_proc.Popen(args_convert,stdout=sub_proc.PIPE,stderr=sub_proc.PIPE)
  dummy = tmp.wait()
  
  #Add another figure
  fig2 = plt.figure(2)
  
  #print MAX_PERCENT_ERR
  #print MIN_PERCENT_ERR  
  
  half_edge = cellSize/2.0
  err_span = MAX_PERCENT_ERR-MIN_PERCENT_ERR
  #ax1 = fig.add_subplot(121)
  numDiv = 100
  ax3 = plt.subplot2grid((numDiv,numDiv),(10,0),rowspan=numDiv,colspan=numDiv-10)
  plt.hold(True)
  for r in rs_list:
    pID_list =  list(position_dict[r]['particles'].keys())
    for pID in pID_list:
      center = position_dict[r]['particles'][pID]['initial']
      percent_error = position_dict[r]['particles'][pID]['percent_error']
      shifted_err_point = percent_error-MIN_PERCENT_ERR
      color_point = shifted_err_point/err_span
      
      plot_color = plt.get_cmap('jet')(color_point)
      #print color_point,' : ',percent_error
      
      xs = [center[0]-half_edge,
	    center[0]-half_edge,
	    center[0]+half_edge,
	    center[0]+half_edge,	      
	    ]
      ys = [center[1]-half_edge,
	    center[1]+half_edge,
	    center[1]+half_edge,
	    center[1]-half_edge,	      
	    ]
	    
      plt.fill(xs,ys,facecolor=plot_color,edgecolor='none',linewidth=0,alpha=1)
      
  plt.xlabel('X position (m)')
  plt.ylabel('Y position (m)')
  plt.title('Percent Error Over Domain:\n' r'$\mathbf{spacing='+str(cellSize)+'}$' ' , ' r'$\mathbf{ppc=(2,2,1)}$' '\n' r'$\mathbf{\nu = '+format(nu,'1.3f')+'}$' ' , ' r'$\mathbf{P_b = '+format(Pb,'1.3e')+'}$')
  corner_text_string = 'Total L2 Error : '+format(tot_L2,'1.3e')+'\nMax analytical displacement:\n'+format(max_analytical_Disp,'1.3e')+' @ r='+format(max_at_r,'1.3f')+'\nScaled L2 Error : '+format(tot_L2/abs(max_analytical_Disp),'1.3e')
  plt.figtext(0.14,0.90,corner_text_string,ha='left',va='top',size='x-small')    
  
  ax4 = fig2.add_subplot(122)
  ax4 = plt.subplot2grid((numDiv,numDiv),(11,numDiv-5),rowspan=numDiv,colspan=4)
  norm = mpl.colors.Normalize(vmin=MIN_PERCENT_ERR, vmax=MAX_PERCENT_ERR)
  cb1 = mpl.colorbar.ColorbarBase(ax4,cmap='jet',norm=norm,orientation='vertical')  

  plt.draw()
  savePNG(img_name+'_ERR',size='1920x1080')
  
  #Convert to smaller image
  #args_convert =   ["convert","1080x768",img_name+'_ERR.png',"-resize","1080x768",'./small/'+img_name+'_ERR'+'_SMALL.png']
  #tmp = sub_proc.Popen(args_convert,stdout=sub_proc.PIPE,stderr=sub_proc.PIPE)
  #dummy = tmp.wait()  
  
  #Track errors
  avg_err = sum(mean_err_percent)/len(mean_err_percent)
  F_err = open(os.path.split(save_path)[0]+'/err_tracking.txt','a+')
  F_err.write('Spacing:'+format(cellSize,'1.4f')+'\tK: '+format(K,'1.4e')+'\tG: '+format(G,'1.4e')+'\tE: '+format(E,'1.4e')+'\tnu: '+format(nu,'1.4e')+'\tTot_L2: '+format(tot_L2,'1.4e')+'\tmax_disp: '+format(max_analytical_Disp,'1.4e')+' @ r: '+format(max_at_r,'1.4f')+'\tscaled_tot_L2: '+format(tot_L2/abs(max_analytical_Disp),'1.4e')+'\tavg error: '+format(avg_err,'2.2f')+' %\n')
  F_err.close()
  
  #plt.show()
