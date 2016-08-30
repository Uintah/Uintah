#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import tempfile
import numpy as np
import subprocess as sub_proc 
 
#Plotting stuff below
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import ticker 

SHOW_ON_MAKE = False
BIG_FIGURE = True
DEFAULT_FORMAT = 'png'

#Useful constants
sqrtTwo = np.sqrt(2.0)
sqrtThree = np.sqrt(3.0)
twoThirds = 2.0/3.0
threeHalves = 3.0/2.0

#Set matplotlib defaults to desired values
if BIG_FIGURE:
  fontSize = 16
  plt.rcParams['font.size']=fontSize
  plt.rcParams['font.weight']='bold'
  plt.rcParams['axes.labelsize']='small'
  plt.rcParams['axes.labelweight']='bold'
  plt.rcParams['legend.fontsize']='small'
else:
  fontSize = 12
  plt.rcParams['axes.titlesize'] = 'medium'
  plt.rcParams['font.size']=fontSize
  plt.rcParams['font.weight']='bold'
  plt.rcParams['axes.labelsize']='small'
  plt.rcParams['axes.labelweight']='bold'
  plt.rcParams['legend.fontsize']='small'
#Set the legend to best fit
plt.rcParams['legend.loc']='best'
#Set linewidth
lineWidth = 2
plt.rcParams['lines.linewidth']=lineWidth
#Set markersize
plt.rcParams['lines.markersize'] = 8
#Set padding for tick labels and size
plt.rcParams['xtick.major.pad']  = 12
plt.rcParams['ytick.major.pad']  = 8
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3

#resolution
plt.rcParams['figure.dpi']=120

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : fontSize}
rc('font', **font) 
rc('text', usetex=True)
 

def saveIMG(name,size='1920x1080',FORMAT=DEFAULT_FORMAT):
  res = float(plt.rcParams['figure.dpi'])
  #Add Check for file already existing as name.png
  if size == '640x480':
    size = [640/res,480/res]
  if size == '1080x768':
    size = [1080/res,768/res]
  if size == '1152x768':
    size = [1152/res,768/res]
  if size == '1280x854':
    size = [1280/res,854/res]    
  if size == '1280x960':
    size = [1280/res,960/res]
  if size == '1920x1080':
    size = [1920/res,1080/res]
  #set the figure size for saving
  plt.gcf().set_size_inches(size[0],size[1])
  #save at speciified resolution
  plt.savefig(name+'.'+FORMAT, bbox_inches=0, dpi=plt.rcParams['figure.dpi'],format=FORMAT)
  #FORMAT = 'png'
  #plt.savefig(name+'.'+FORMAT, bbox_inches=0, dpi=plt.rcParams['figure.dpi'],format=FORMAT)

def str_to_mathbf(string):
  #Only works with single spaces no leading space
  string = string.split()
  return_string = ''
  for elem in string:
    elem = r'$\mathbf{'+elem+'}$'
    return_string+=elem+'  '
  return return_string[0:-1]
  

def sign(x,y):
  if y>=0:
    return abs(x)
  else:
    return -abs(x)

def sigma_iso(sigma):
  return (np.trace(sigma)/3.0)*np.eye(3)
  
def sigma_dev(sigma):
  return sigma-sigma_iso(sigma)

def sigma_I1(sigma):
  return sigma.trace()
  
def sigma_J2(sigma):
  return 0.5*np.dot(sigma_dev(sigma),sigma_dev(sigma)).trace()

def sigma_J3(sigma):
  return (1/3.0)*np.dot(np.dot(sigma_dev(sigma),sigma_dev(sigma)),sigma_dev(sigma)).trace()

def sigma_mag(sigma):
  #Returns the magnitude of a second-rank tensor
  #return np.linalg.norm(sigma)
  return np.sqrt(DblDot(sigma,sigma))

def DblDot(x,y):#Returns the double inner product of two second-rank tensors
  val=0
  for i in range(0,3):
      for j in range(0,3):
	  val=val+(x[i][j]*y[i][j])
  return val

def sigma_tau(sigma):
  #return sign(np.sqrt(sigma_J2(sigma)),sigma_J3(sigma))
  return sign(np.sqrt(sigma_J2(sigma)),sigma_J3(sigma))

def get_ps_and_qs(sigmas):
  ps = []
  qs = []
  for sigma in sigmas:
    qs.append(sign(sqrtThree*np.sqrt(sigma_J2(sigma)),sigma_J3(sigma)))
    ps.append(sigma_I1(sigma)/3.0)
  return np.array(ps),np.array(qs)
  
def get_pStress(uda_path):
  NAN_FAIL = False
  #Extract stress history
  print "Extracting stress history..."
  args = ["partextract","-partvar","p.stress",uda_path]
  F_stress = tempfile.TemporaryFile()
  #F_stress = open("./tempStressFileOut.txt","w+")
  #open(os.path.split(uda_path)[0]+'/stressHistory.dat',"w+")
  tmp = sub_proc.Popen(args,stdout=F_stress,stderr=sub_proc.PIPE)
  dummy = tmp.wait()
  print('Done.')
  #Read file back in
  F_stress.seek(0)
  times = []
  sigmas = []
  for line in F_stress:
    line = line.strip().split()
    times.append(float(line[0]))
    S11 = np.float64(line[4])
    S12 = np.float64(line[5])
    S13 = np.float64(line[6])
    S21 = np.float64(line[7])
    S22 = np.float64(line[8])
    S23 = np.float64(line[9])
    S31 = np.float64(line[10])
    S32 = np.float64(line[11])
    S33 = np.float64(line[12])
    sigmas.append(np.array([[S11,S12,S13],[S21,S22,S23],[S31,S32,S33]]))
    for i in range(3):
      for j in range(3):
	if np.isnan(sigmas[-1][i][j]):
	  NAN_FAIL = True
  F_stress.close()
  if NAN_FAIL:
    print "\nERROR: 'nan's found reading in stress. Will not plot correctly"
  return times,sigmas

def get_pDeformationMeasure(uda_path):
  NAN_FAIL = False
  #Extract stress history
  print "Extracting deformation history..."
  args = ["partextract","-partvar","p.deformationMeasure",uda_path]
  F_defMes = tempfile.TemporaryFile()
  #open(os.path.split(uda_path)[0]+'/stressHistory.dat',"w+")
  tmp = sub_proc.Popen(args,stdout=F_defMes,stderr=sub_proc.PIPE)
  dummy = tmp.wait()
  print('Done.')
  #Read file back in
  F_defMes.seek(0)
  times = []
  Fs = []
  for line in F_defMes:
    line = line.strip().split()
    times.append(float(line[0]))
    F11 = np.float64(line[4])
    F12 = np.float64(line[5])
    F13 = np.float64(line[6])
    F21 = np.float64(line[7])
    F22 = np.float64(line[8])
    F23 = np.float64(line[9])
    F31 = np.float64(line[10])
    F32 = np.float64(line[11])
    F33 = np.float64(line[12])
    Fs.append(np.array([[F11,F12,F13],[F21,F22,F23],[F31,F32,F33]]))
    for i in range(3):
      for j in range(3):
	if np.isnan(Fs[-1][i][j]):
	  NAN_FAIL = True
  F_defMes.close()
  if NAN_FAIL:
    print "\nERROR: 'nan's found reading in stress. Will not plot correctly"
  return times,Fs

def get_epsilons(uda_path):
  #Assumes no shear strains
  times,Fs = get_pDeformationMeasure(uda_path)
  epsils = []
  for F in Fs:
    epsils.append(np.array([[np.log(F[0][0]),0,0],[0,np.log(F[1][1]),0],[0,0,np.log(F[2][2])]]))
  return times,epsils

def get_pKappa(uda_path):
  #Extract stress history
  print "Extracting kappa history..."
  args = ["partextract","-partvar","p.kappa",uda_path]
  F_kappa = tempfile.TemporaryFile()
  #open(os.path.split(uda_path)[0]+'/kappaHistory.dat',"w+")
  tmp = sub_proc.Popen(args,stdout=F_kappa,stderr=sub_proc.PIPE)
  dummy = tmp.wait()
  print('Done.')
  #Read file back in
  F_kappa.seek(0)
  times = []
  kappas = []
  for line in F_kappa:
    line = line.strip().split()
    times.append(float(line[0]))
    kappas.append(float(line[4]))
  F_kappa.close()
  return times,kappas

def get_pPlasticStrainVol(uda_path):
  FAIL_NAN = False
  #Extract stress history
  print "Extracting plasticStrainVol history..."
  args = ["partextract","-partvar","p.evp",uda_path]
  F_plasticStrainVol = tempfile.TemporaryFile()
  #open(os.path.split(uda_path)[0]+'/plasticStrainVolHistory.dat',"w+")
  tmp = sub_proc.Popen(args,stdout=F_plasticStrainVol,stderr=sub_proc.PIPE)
  dummy = tmp.wait()
  print('Done.')
  #Read file back in
  F_plasticStrainVol.seek(0)
  times = []
  plasticStrainVol = []
  for line in F_plasticStrainVol:
    line = line.strip().split()
    times.append(float(line[0]))
    plasticStrainVol.append(np.float64(line[4]))
    if np.isnan(plasticStrainVol[-1]):
      FAIL_NAN = True
  F_plasticStrainVol.close()
  if FAIL_NAN:
    print "\ERROR: 'nan' encountered while retrieving p.evp, will not plot correctly."  
  return times,plasticStrainVol  

def get_pElasticStrainVol(uda_path):
  FAIL_NAN = False
  #Extract elastic strain history
  print "Extracting elasticStrainVol history..."
  args = ["partextract","-partvar","p.eve",uda_path]
  F_elasticStrainVol = tempfile.TemporaryFile()
  #open(os.path.split(uda_path)[0]+'/elasticStrainVolHistory.dat',"w+")
  tmp = sub_proc.Popen(args,stdout=F_elasticStrainVol,stderr=sub_proc.PIPE)
  dummy = tmp.wait()
  print('Done.')
  #Read file back in
  F_elasticStrainVol.seek(0)
  times = []
  elasticStrainVol = []
  for line in F_elasticStrainVol:
    line = line.strip().split()
    times.append(float(line[0]))
    elasticStrainVol.append(np.float64(line[4]))
    if np.isnan(elasticStrainVol[-1]):
      FAIL_NAN = True
  F_elasticStrainVol.close()
  if FAIL_NAN:
    print "\ERROR: 'nan' encountered while retrieving p.eve, will not plot correctly."
  return times,elasticStrainVol  

def get_totalStrainVol(uda_path):
  times,plasticStrainVol = get_pPlasticStrainVol(uda_path)
  times,elasticStrainVol = get_pElasticStrainVol(uda_path)
  
  print 'num plastic : ',len(plasticStrainVol)
  print 'num elastic : ',len(elasticStrainVol)

  totalStrainVol = np.array(plasticStrainVol)+np.array(elasticStrainVol)
  return times,totalStrainVol

def get_defTable(uda_path,working_dir):
  #Determine the defTable file
  try:
    ups_file = os.path.abspath(uda_path)+'/input.xml.orig'
    F = open(ups_file,"r")
  except:
    ups_file = os.path.abspath(uda_path)+'/input.xml'
    F = open(ups_file,"r")
  for line in F:
    if '<PrescribedDeformationFile>' in line and '</PrescribedDeformationFile>' in line:
      def_file = line.split('<PrescribedDeformationFile>')[1].split('</PrescribedDeformationFile>')[0].strip()
  F.close()
  #Assumes the input deck and uda share the same parent folder. 
  def_file = working_dir+'/'+def_file
  F = open(def_file,'r')
  times = []
  Fs = []
  for line in F:
    line = line.strip().split()    
    if line:
      times.append(float(line[0]))
      Fs.append(np.array([[float(line[1]),float(line[2]),float(line[3])],
			[float(line[4]),float(line[5]),float(line[6])],
			[float(line[7]),float(line[8]),float(line[9])]]))
  F.close()
  return times,Fs 

def exp_fmt(x,loc):
  tmp = format(x,'1.2e').split('e')
  lead = tmp[0]
  exp = str(int(tmp[1]))
  if exp=='0' and lead=='0.00':
    return r'$\mathbf{0.00}$'
  else:
    if int(exp)<10 and int(exp)>0:
      exp = '+0'+exp
    elif int(exp)>-10 and int(exp)<0:
      exp = '-0'+exp.split('-')[1]
    elif int(exp)>10:
      exp = '+'+exp
    return r'$\mathbf{'+lead+r'x{}10^{'+exp+'}}$' 

def eqShear_vs_meanStress(xs,ys,Xlims=False,Ylims=False,LINE_LABEL='Uintah',GRID=True):

  ax1 = plt.subplot(111)
  plt.plot(np.array(xs),np.array(ys),'-r',label=LINE_LABEL)
  
  plt.xlabel(str_to_mathbf('Mean Stress, p (Pa)'))
  plt.ylabel(str_to_mathbf('Equivalent Shear Stress, q, (Pa)'))
  
  ax1.xaxis.set_major_formatter(formatter_exp)
  ax1.yaxis.set_major_formatter(formatter_exp)    

  if Xlims:
    ax1.set_xlim(Xlims[0],Xlims[1])
  if Ylims:
    ax1.set_ylim(Ylims[0],Ylims[1])
  if GRID:    
    plt.grid(True)  

  return ax1

def get_yield_surface(uda_path):
  #Reads in FSLOPE, FSLOPE_p, PEAKI1, CR, and P0
  #WILL ONLY WORK FOR SINGLE ELEMENT TESTS OR DECKS 
  #HAVING ONLY ONE ARENISCA SPECIFICATION
  try:
    ups_file = os.path.abspath(uda_path)+'/input.xml.orig'
    F_ups = open(ups_file,"r")
  except:
    ups_file = os.path.abspath(uda_path)+'/input.xml'
    F_ups = open(ups_file,"r")
  check_lines = False
  already_read = False
  material_dict = {}
  for line in F_ups:
    if '<constitutive_model' in line and 'type' in line and '"Arenisca"' in line and not(already_read):
      check_lines = True
    if check_lines and not(already_read):
      if '<B0>' in line:
        material_dict['B0'] = float(line.split('<B0>')[1].split('</B0>')[0].strip())
      if '<G0>' in line:
        material_dict['G0'] = float(line.split('<G0>')[1].split('</G0>')[0].strip())
      if '<hardening_modulus>' in line:
        material_dict['hardening_modulus'] = float(line.split('<hardening_modulus>')[1].split('</hardening_modulus>')[0].strip())        
      if '<FSLOPE>' in line:
        material_dict['FSLOPE'] = float(line.split('<FSLOPE>')[1].split('</FSLOPE>')[0].strip())
      if '<FSLOPE_p>' in line:
        material_dict['FSLOPE_p'] = float(line.split('<FSLOPE_p>')[1].split('</FSLOPE_p>')[0].strip())        
      if '<PEAKI1>' in line:
        material_dict['PEAKI1'] = float(line.split('<PEAKI1>')[1].split('</PEAKI1>')[0].strip())      
      if '<CR>' in line:
        material_dict['CR'] = float(line.split('<CR>')[1].split('</CR>')[0].strip())
      if '<p0_crush_curve>' in line:
        material_dict['P0'] = float(line.split('<p0_crush_curve>')[1].split('</p0_crush_curve>')[0].strip())
      if '<p1_crush_curve>' in line:
        material_dict['P1'] = float(line.split('<p1_crush_curve>')[1].split('</p1_crush_curve>')[0].strip())        
      if '<p3_crush_curve>' in line:
        material_dict['P3'] = float(line.split('<p3_crush_curve>')[1].split('</p3_crush_curve>')[0].strip())
      if '<p4_fluid_effect>' in line:
        material_dict['P4']  = float(line.split('<p4_fluid_effect>')[1].split('</p4_fluid_effect>')[0].strip())        
      if '<fluid_B0>' in line:
        material_dict['fluid_B0']  = float(line.split('<fluid_B0>')[1].split('</fluid_B0>')[0].strip())        
      if '<fluid_pressure_initial>' in line:
        material_dict['P_f0']  = float(line.split('<fluid_pressure_initial>')[1].split('</fluid_pressure_initial>')[0].strip())                      
      if '<subcycling_characteristic_number>' in line:
        material_dict['subcycling char num']  = float(line.split('<subcycling_characteristic_number>')[1].split('</subcycling_characteristic_number>')[0].strip())                
      if '<kinematic_hardening_constant>' in line:
        material_dict['hardening_constant']  = float(line.split('<kinematic_hardening_constant>')[1].split('</kinematic_hardening_constant>')[0].strip())
      if '<T1_rate_dependence>' in line:
	material_dict['T1']  = float(line.split('<T1_rate_dependence>')[1].split('</T1_rate_dependence>')[0].strip())
      if '<T2_rate_dependence>' in line:
	material_dict['T2']  = float(line.split('<T2_rate_dependence>')[1].split('</T2_rate_dependence>')[0].strip())
      if '<gruneisen_parameter>' in line:
	material_dict['gruneisen_parameter']  = float(line.split('<gruneisen_parameter>')[1].split('</gruneisen_parameter>')[0].strip())
      if '</constitutive_model>' in line:
	already_read = True
	check_lines = False
  F_ups.close()
  PRINTOUT = False
  if PRINTOUT:
    print '--Material Specification--'
    for key in material_dict:
      print key,':',material_dict[key]
      
  #tmp_string = r'$\mathbf{\underline{Material}}$'+' '+r'$\mathbf{\underline{Properties:}}$'+'\n'
  tmp_string = r'$\mathbf{\underline{Material\phantom{1}Properties:}}$'+'\n'
  key_list = material_dict.keys()
  key_list.sort()
  for key in key_list:
    if '_' in key:
      tmp = key.split('_')
      tmp = str_to_mathbf(tmp[0]+'_'+'{'+tmp[1]+'}')
      tmp_string += tmp+str_to_mathbf(' = ')+str_to_mathbf(format(material_dict[key],'1.3e'))+'\n'
    else:
      tmp = key
      if key == 'subcycling char num':
	tmp_string += str_to_mathbf(tmp+' = '+format(material_dict[key],'4.1f'))+'\n'
      else:
        tmp_string += str_to_mathbf(tmp+' = '+format(material_dict[key],'1.3e'))+'\n'
  material_dict['material string'] = tmp_string[0:-1]
  if PRINTOUT:
    print tmp_string
  return material_dict

def get_kappa(PEAKI1,P0,FSLOPE_p,CR):
  PEAKI1,P0,FSLOPE_p,CR
  numerator = P0+(FSLOPE_p*CR*PEAKI1)
  denomerator = 1.0+(FSLOPE_p*CR)
  kappa = numerator/denomerator
  return kappa

def get_rs(nPoints,FSLOPE_p,PEAKI1,P0,CR):

  kappa = get_kappa(PEAKI1,P0,FSLOPE_p,CR)  
  I1s = np.linspace(PEAKI1,P0,nPoints)
  rs = []
  for I1 in I1s:
    inner_root = (1.0-(pow(kappa-I1,2.0)/pow(kappa-P0,2.0)))
    #print inner_root
    #print kappa-I1
    #print 'kappa:',kappa,'\tI1:',I1
    r = FSLOPE_p*(I1-PEAKI1)*np.sqrt(2.0*inner_root)
    rs.append(r)
  return I1s,rs
  
def I1_to_zbar(I1s):
  sqrt_3 = np.sqrt(3.0)
  if type(I1s) in [list,np.ndarray]:
    zbars = []
    for I1 in I1s:
      zbars.append(-I1/sqrt_3)
    return zbars
  elif type(I1s) in [int,float,np.float64]:
    return -I1s/sqrt_3
  else:
    print '\nERROR: cannot compute zbar from I1. Invalid type.\n\ttype(I1)\t:\t',type(I1s)
    return None

def get_porosity(I1,material_dict):
  P0 = material_dict['P0']
  P1 = material_dict['P1']
  P3 = material_dict['P3']
  if I1<P0:
    porosity =  1-np.exp(-P3*np.exp(P1*(I1-P0)))
  else:
    porosity = 1-np.exp(-(I1/P0)**(P0*P1*P3)-P3+1)
  return porosity

def plot_crush_curve(uda_path,I1lims=[-10000,0]):
  nPoints = 500
  material_dict = get_yield_surface(uda_path)
  P0 = material_dict['P0']
  I1sC = np.linspace(I1lims[0],P0,nPoints)
  I1sT = np.linspace(P0,I1lims[1],nPoints)
  
  porosityCs = []
  porosityTs = []
  for I1 in I1sC:
    porosityCs.append(get_porosity(I1,material_dict))
  for I1 in I1sT:
    porosityTs.append(get_porosity(I1,material_dict))
  
  plt.hold(True)
  plt.plot(I1sC,porosityCs,'--g',linewidth=lineWidth+1,label='Analytical crush curve - Compression')
  plt.hold(True)
  plt.plot(I1sT,porosityTs,'--b',linewidth=lineWidth+1,label='Analytical crush curve - Tension')

 
def plot_yield_surface_OLD(uda_path):
  nPoints = 500
  material_dict = get_yield_surface(uda_path)
  FSLOPE = material_dict['FSLOPE']
  FSLOPE_p = material_dict['FSLOPE_p']
  PEAKI1 = material_dict['PEAKI1']
  CR = material_dict['CR']
  P0 = material_dict['P0']
  I1s,rs = get_rs(nPoints,FSLOPE_p,PEAKI1,P0,CR)
  zbars = I1_to_zbar(I1s)
  #WTF?
  for i in range(len(rs)):
    rs[i] = -rs[i]  
  #print zbars
  #print rs
  plt.plot(np.array(I1s)/3.0,rs,'--k',linewidth=lineWidth+1,label='Initial Yield Surface')
  plt.plot(np.array(I1s)/3.0,-np.array(rs),'--k',linewidth=lineWidth+1)    

def dyadicMultiply(a,b):
  T = np.eye(3)
  for i in range(3):
    for j in range(3):
      T[i][j] = a[i]*b[j]
  return T

def tensorSpectralMath(A,function,arg=None):
  #Get eigen values and associated eigenvectors
  eigVals,eigVecs = np.linalg.eig(A)
  #Find repeated roots conditions
  if eigVals[0] == eigVals[1] and eigVals[1] != eigVals[2]:
    OneTwoRepeated = True
    TwoThreeRepeated = False
    AllEqual = False
  elif eigVals[0] != eigVals[1] and eigVals[1] == eigVals[2]:
    OneTwoRepeated = False
    TwoThreeRepeated = True
    AllEqual = False      
  elif eigVals[0] == eigVals[1] and eigVals[0] == eigVals[2]:
    OneTwoRepeated = False
    TwoThreeRepeated = False
    AllEqual = True
  else:
    OneTwoRepeated = False
    TwoThreeRepeated = False
    AllEqual = False
  #Use spectral decomposition to compute the log of Epsilon to get F
  modA = np.zeros((3,3))
  #First two roots repeat
  if OneTwoRepeated:
    #Construct basis tensors
    P01 = (dyadicMultiply(eigVecs[0],eigVecs[0])+dyadicMultiply(eigVecs[1],eigVecs[1]))
    P22 = dyadicMultiply(eigVecs[2],eigVecs[2])
    if function == 'log':
      modA += np.log(eigVals[1])*P01    
      modA += np.log(eigVals[2])*P22     
    elif function == 'exp':
      modA += np.exp(eigVals[1])*P01    
      modA += np.exp(eigVals[2])*P22       
    elif function == 'pow':
      modA += np.power(eigVals[1],arg)*P01    
      modA += np.power(eigVals[2],arg)*P22
  #Second two roots repeat
  elif TwoThreeRepeated:
    #Construct basis tensors
    P00 = dyadicMultiply(eigVecs[0],eigVecs[0])
    P12 = (dyadicMultiply(eigVecs[1],eigVecs[1])+dyadicMultiply(eigVecs[2],eigVecs[2]))
    if function == 'log':
      modA += np.log(eigVals[0])*P00
      modA += np.log(eigVals[1])*P12  
    elif function == 'exp':
      modA += np.exp(eigVals[0])*P00
      modA += np.exp(eigVals[1])*P12     
    elif function == 'pow':
      modA += np.power(eigVals[0],arg)*P00
      modA += np.power(eigVals[1],arg)*P12
  #All are repeated roots
  elif AllEqual:
    #Construct basis tensor
    P = (dyadicMultiply(eigVecs[0],eigVecs[0])+dyadicMultiply(eigVecs[1],eigVecs[1])+dyadicMultiply(eigVecs[2],eigVecs[2]))
    if function == 'log':
      modA += np.log(eigVals[0])*P
    elif function == 'exp':
      modA += np.exp(eigVals[0])*P     
    elif function == 'pow':
      modA += np.power(eigVals[0],arg)*P
  #No repeated roots
  else:
    for i in range(3):
      if function == 'log':
	modA += np.log(eigVals[i])*dyadicMultiply(eigVecs[i],eigVecs[i])
      elif function == 'exp':
	modA += np.exp(eigVals[i])*dyadicMultiply(eigVecs[i],eigVecs[i])  
      elif function == 'pow':
	modA += np.power(eigVals[i],arg)*dyadicMultiply(eigVecs[i],eigVecs[i])
  return modA  

def tensor_exp(A):
  return tensorSpectralMath(A,'exp',arg=None)
def tensor_pow(A,ARG):
  return tensorSpectralMath(A,'pow',ARG)
def tensor_log(A):
  return tensorSpectralMath(A,'log')
  
def J2VM(epsil_dot,dt,sig_Beg,K,G,tau_y):
  #J2 plasticity Von misses material model for 3D
  #Inputs: epsil_dot, dt, sig_Beg, K, G, tau_y
  #Outputs: epsil_Elastic_dot, epsil_Plastic_dot, sig_End
  #Initialize the trial stress state
  sig_Trial = sig_Beg+((2.0*G*sigma_dev(epsil_dot))+3.0*K*sigma_iso(epsil_dot))*dt
  #Determine if this is below, on, or above the yeild surface
  test = sigma_mag(sigma_dev(sig_Trial))/(np.sqrt(2.0)*tau_y)
  if test<=1:
      #Stress state is elastic
      sig_End = sig_Trial
      epsil_Plastic_dot = np.zeros((3,3))
      epsil_Elastic_dot = epsil_dot
  elif test>1:
      #Stress state elastic-plastic
      sig_End = (sigma_dev(sig_Trial)/test)+sigma_iso(sig_Trial)
      #Evaluate the consistent stress rate
      sig_dot = (sig_End-sig_Beg)/dt
      #Apply hookes law to get the elastic strain rate
      epsil_Elastic_dot = sigma_dev(sig_dot)/(2.0*G) + sigma_iso(sig_dot)/(3.0*K)
      #Apply strain rate decomposition relationship to get plastic strain rate
      epsil_Plastic_dot = epsil_dot-epsil_Elastic_dot
  #Determine the equivalent stress and equivalent plastic strain rate
  #sig_Eq = np.sqrt(3/2)*sigma_mag(sigma_dev(sig_End))
  #epsil_Plastic_dot_Eq = np.sqrt(3/2)*sigma_mag(sigma_dev(epsil_Plastic_dot))
  #ans={'Elastic dot':epsil_Elastic_dot,'Plastic dot':epsil_Plastic_dot,'Stress State':sig_End}
  return sig_End    

def defTable_to_J2Solution(def_times,Fs,bulk_mod,shear_mod,tau_yield,num_substeps=1000):
  #Assumes: 
  
  print 'Solving for analytical solution...'
  analytical_epsils = [np.array([[0,0,0],[0,0,0],[0,0,0]])]
  analytical_sigmas = [np.array([[0,0,0],[0,0,0],[0,0,0]])]
  analytical_times = [def_times[0]]
  
  epsils = []
  for F in Fs:
    epsils.append(tensor_log(F))
    
  for leg in range(len(def_times)-1):
    t_start = def_times[leg]
    leg_delT = def_times[leg+1]-t_start
    leg_sub_delT = float(leg_delT)/float(num_substeps)
    leg_del_epsil = (epsils[leg+1]-epsils[leg])
    leg_epsil_dot = leg_del_epsil/leg_delT
    for i in range(num_substeps):
      t_now = t_start+float(i)*leg_sub_delT
      analytical_times.append(t_now)
      analytical_sigmas.append(J2VM(leg_epsil_dot,leg_sub_delT,analytical_sigmas[-1],bulk_mod,shear_mod,tau_yield))
      analytical_epsils.append(analytical_epsils[-1]+(leg_epsil_dot*leg_sub_delT))
  analytical_epsils.append(analytical_epsils[-1]+(leg_epsil_dot*leg_sub_delT))   
  analytical_sigmas.append(J2VM(leg_epsil_dot,leg_sub_delT,analytical_sigmas[-1],bulk_mod,shear_mod,tau_yield))
  analytical_times.append(def_times[-1])
  print 'Done.'
  return analytical_times,analytical_sigmas,analytical_epsils

def J2_at_Yield(uda_path):
  material_dict = get_yield_surface(uda_path)
  B0 = material_dict['B0']
  G0 = material_dict['G0']
  hardenig_mod = material_dict['hardening_modulus']
  FSLOPE = material_dict['FSLOPE']
  FSLOPE_p = material_dict['FSLOPE_p']
  PEAKI1 = material_dict['PEAKI1']
  CR = material_dict['CR']
  P0 = material_dict['P0']
  P1 = material_dict['P1']
  P3 = material_dict['P3']
  P4 = material_dict['P4']
  fluid_B0 = material_dict['fluid_B0']
  Pf0 = material_dict['P_f0']
  subcyc_char_num = material_dict['subcycling char num']
  hardening_const = material_dict['hardening_constant']
  
  # homel: modified for elliptical cap
  # kappa_initial = (P0+FSLOPE*CR*PEAKI1)/(1.0+FSLOPE*CR)
  kappa_initial = P0-(FSLOPE*P0)/np.sqrt(CR**2+FSLOPE**2)+(FSLOPE*PEAKI1)/np.sqrt(CR**2+FSLOPE**2)
  I1 = 0
  I1_plus3Pf0 = I1+3.0*Pf0
  if I1_plus3Pf0 >= kappa_initial and I1_plus3Pf0<= PEAKI1:
    J2 = (FSLOPE**2)*((I1-PEAKI1+3.0*Pf0)**2)
  elif I1_plus3Pf0 >= P0 and I1_plus3Pf0 < kappa_initial:
    J2 = ((FSLOPE**2)*((I1-PEAKI1+3.0*Pf0)**2))*(1.0-((I1+CR*FSLOPE*I1-P0-CR*FSLOPE*PEAKI1+3.0*Pf0+3.0*CR*FSLOPE*Pf0)**2/((CR**2)*(FSLOPE**2)*(P0-PEAKI1)**2)))
  else:
    J2 = 0.0  
  return J2
  
def plot_yield_surface(uda_path,PLOT_TYPE='J2_vs_I1'):
  num_points = 500
  material_dict = get_yield_surface(uda_path)
  B0 = material_dict['B0']
  G0 = material_dict['G0']
  hardenig_mod = material_dict['hardening_modulus']
  FSLOPE = material_dict['FSLOPE']
  FSLOPE_p = material_dict['FSLOPE_p']
  PEAKI1 = material_dict['PEAKI1']
  CR = material_dict['CR']
  P0 = material_dict['P0']
  P1 = material_dict['P1']
  P3 = material_dict['P3']
  P4 = material_dict['P4']
  fluid_B0 = material_dict['fluid_B0']
  Pf0 = material_dict['P_f0']

  #homel: modified for elliptical cap  
  #kappa_initial = (P0+FSLOPE*CR*PEAKI1)/(1.0+FSLOPE*CR)
  beta = FSLOPE*((3.0*B0)/(2.0*G0))*np.sqrt(6.0)
  apex_initial = (P0*CR**2 + beta*(-beta + np.sqrt(beta**2 + CR**2))*(-P0 + PEAKI1)) / (CR**2)
  ellipse_a = apex_initial-P0
  ellipse_b = CR*ellipse_a*(2.0*G0)/(3.0*B0)/np.sqrt(6.0)
  kappa_initial = P0-(beta*P0)/np.sqrt(CR**2+beta**2)+(beta*PEAKI1)/np.sqrt(CR**2+beta**2)
  I1s = np.linspace(P0-3.0*Pf0,PEAKI1-3.0*Pf0,num_points)
  
  #print 'Region 1:: ','I1 >= kappa initial-3.0*Pf0 : ',kappa_initial-3.0*Pf0,' ','I1 <= PEAKI1-3*Pf0 : ',PEAKI1-3.0*Pf0
  #print 'Region 2:: ','I1 >= P0-3*Pf0 : ',P0-3.0*Pf0,' ','I1 < kappa_initial-3*Pf0 : ',kappa_initial-3.0*Pf0
  #print 'Region 3:: Not Region 1 or 2'

  #J2 versus I1
  J2s = []
  PLOT = True
  for I1 in I1s:
    I1_plus3Pf0 = I1+3.0*Pf0
    if I1_plus3Pf0 >= kappa_initial and I1_plus3Pf0<= PEAKI1:
      J2 = (FSLOPE**2)*((I1-PEAKI1+3.0*Pf0)**2)
    elif I1_plus3Pf0 >= P0 and I1_plus3Pf0 < kappa_initial:
# homel: modified for elliptical cap.
      J2=(ellipse_b**2)*(1.0-((I1-apex_initial)**2)/(ellipse_a**2))
#      J2 =((np.sqrt(I1-P0)*np.sqrt((FSLOPE*(P0-PEAKI1)*(-2.0*(FSLOPE**2)*(-FSLOPE+np.sqrt(CR**2+FSLOPE**2))*(P0-PEAKI1)+(CR**2)*(np.sqrt(CR**2+FSLOPE**2)*I1+2.0*FSLOPE*P0-np.sqrt(CR**2+FSLOPE**2)*P0-2.0*FSLOPE*PEAKI1)))/(CR**2+FSLOPE**2)))/np.sqrt((FSLOPE*(-P0+PEAKI1))/np.sqrt(CR**2+FSLOPE**2)))**2
#      J2 = ((FSLOPE**2)*((I1-PEAKI1+3.0*Pf0)**2))*(1.0-((I1+CR*FSLOPE*I1-P0-CR*FSLOPE*PEAKI1+3.0*Pf0+3.0*CR*FSLOPE*Pf0)**2/((CR**2)*(FSLOPE**2)*(P0-PEAKI1)**2)))
    else:
      J2 = 0.0
    J2s.append(J2)
  
  if PLOT_TYPE == 'J2_vs_I1':
    xs = I1s
    ys = np.array(J2s)  
  elif PLOT_TYPE == 'sqrtJ2_vs_I1':
    xs = I1s
    ys = np.sqrt(np.array(J2s))   
  elif PLOT_TYPE == 'r_vs_z':
    xs = np.array(I1s)/np.sqrt(3.0)
    ys = np.sqrt(2.0*np.array(J2s))
  elif PLOT_TYPE == 'q_vs_I1':
    xs = I1s
    ys = np.sqrt(3.0*np.array(J2s))
  elif PLOT_TYPE == 'q_vs_p':
    xs = np.array(I1s)/3.0
    ys = np.sqrt(3.0*np.array(J2s))
  else:
    PLOT = False
    print '\nError: invalid plot type specified for initial yield surface plot.\n\tPLOT_TYPE:',PLOT_TYPE
  if PLOT:
    plt.plot(xs,ys,'--k',linewidth=lineWidth+1,label='Initial Yield Surface')
    plt.plot(xs,-ys,'--k',linewidth=lineWidth+1)  

def test_yield_surface(uda_path):
  plot_yield_surface_2(uda_path,'J2_vs_I1')
  plt.show()
  plot_yield_surface_2(uda_path,'sqrtJ2_vs_I1')
  plt.show()
  plot_yield_surface_2(uda_path,'r_vs_z')
  plt.show()
  plot_yield_surface_2(uda_path,'q_vs_I1')
  plt.show()
  plot_yield_surface_2(uda_path,'q_vs_p')
  plt.show()
  

### ----------
#	Test Methods Below
### ----------

formatter_int = ticker.FormatStrFormatter('$\mathbf{%g}$')
formatter_exp = ticker.FuncFormatter(exp_fmt)

def test01_postProc(uda_path,save_path,**kwargs):
  print "Post Processing Test: 01 - Uniaxial Compression With Rotation"
  times,sigmas = get_pStress(uda_path)
  material_dict = get_yield_surface(uda_path)
  Sxx = []
  Syy = []
  for sigma in sigmas:
    Sxx.append(sigma[0][0])
    Syy.append(sigma[1][1])    
  
  ###PLOTTING
  plt.figure(1)
  plt.clf()
  
  if BIG_FIGURE:
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
    plt.subplots_adjust(right=0.75)  
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)  
    

  #Syy
  ax2 = plt.subplot(212)
  #without rotation
  plt.plot([0,1],[0,0],'-b')
  #simulation results
  plt.plot(times,Syy,'-r')  
  #guide line
  plt.plot([0,1],[0,-60],'--g')
  #labels and limits
  ax2.set_ylim(-70,10)
  plt.grid(True)
  ax2.xaxis.set_major_formatter(formatter_int)
  ax2.yaxis.set_major_formatter(formatter_int)
  plt.ylabel(str_to_mathbf('\sigma_{yy} (Pa)'))
  plt.xlabel(str_to_mathbf('Time (s)'))
  #Sxx
  ax1 = plt.subplot(211,sharex=ax2,sharey=ax2)
  plt.setp(ax1.get_xticklabels(), visible=False)
  #without rotation
  plt.plot([0,1],[0,-60],'-b',label='No rotation') 
  #simulation results
  plt.plot(times,Sxx,'-r',label='Uintah')
  #guide lines
  plt.plot([0,1],[0,0],'--g',label='Guide lines')  
  #labels
  ax1.set_ylim(-70,10)
  plt.grid(True)
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)
  ax1.set_yticks([0,-20,-40,-60])  
  plt.ylabel(str_to_mathbf('\sigma_{xx} (Pa)')) 
  plt.legend(loc=3)
  
  if BIG_FIGURE:
    plt.title('AreniscaTest 01:\nUniaxial Compression With Rotation')
    saveIMG(save_path+'/Test01_verificationPlot','1280x960')
  else:
    saveIMG(save_path+'/Test01_verificationPlot','640x480')

  if SHOW_ON_MAKE:
    plt.show()
  
def test02_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 02 - Vertex Treatment"
  times,sigmas = get_pStress(uda_path)
  ps,qs = get_ps_and_qs(sigmas)
  rs = qs*np.sqrt(2.0/3.0)
  zs = (ps*3.0)/np.sqrt(3.0)
  Sxx = []
  Syy = []
  Szz = []
  for sigma in sigmas:
    Sxx.append(sigma[0][0])
    Syy.append(sigma[1][1])
    Szz.append(sigma[2][2])

  #Analytical Solutions
  #Drucker-Prager constants
  r0 = 50.0
  z0 = 50.0*sqrtThree
  #Solution From Brannon Leelavanichkul paper
  analytical_times = [0,1,threeHalves,2.0,5.0/2.0,3.0]
  analytical_S11 = np.array([0,
			    -850.0/3.0,
			     (-50.0/3.0)*(9.0+4.0*np.sqrt(6.0)),
			     (-50.0/3.0)*(9.0+4.0*np.sqrt(6.0)),
			     (50.0/3.0)*(2.0*np.sqrt(6)-3.0),
			     160.0*np.sqrt(twoThirds)-110.0
			     ])
  analytical_S22 = np.array([0,
			    -850.0/3.0,
			    (50.0/3.0)*(2.0*np.sqrt(6.0)-9.0),
			    (50.0/3.0)*(2.0*np.sqrt(6.0)-9.0),
			    (-50.0/3.0)*(3.0+np.sqrt(6.0)),
			    (-10.0/3.0)*(33.0+8.0*np.sqrt(6.0))
			    ])
  analytical_S33 = np.array([0,
			    -850.0/3.0,
			    (50.0/3.0)*(2.0*np.sqrt(6.0)-9.0),
			    (50.0/3.0)*(2.0*np.sqrt(6.0)-9.0),
			    (-50.0/3.0)*(3.0+np.sqrt(6.0)),
			    (-10.0/3.0)*(33.0+8.0*np.sqrt(6.0))
			    ])

  analytical_mean = (analytical_S11+analytical_S22+analytical_S33)/3.0
  analytical_I1 = analytical_S11+analytical_S22+analytical_S33
  sigIso = (1.0/3.0)*analytical_I1
  
  analytical_sigDev1 = analytical_S11-sigIso
  analytical_sigDev2 = analytical_S22-sigIso
  analytical_sigDev3 = analytical_S33-sigIso
  
  analytical_J2 = (1.0/2.0)*(pow(analytical_sigDev1,2)+pow(analytical_sigDev2,2)+pow(analytical_sigDev3,2))
  analytical_J3 = (1.0/3.0)*(pow(analytical_sigDev1,3)+pow(analytical_sigDev2,3)+pow(analytical_sigDev3,3))
  analytical_z = analytical_I1/sqrtThree
  analytical_q = []
  analytical_r = []
  
  for idx,J2 in enumerate(analytical_J2):
    J3 = analytical_J3[idx]
    analytical_q.append(sign(sqrtThree*np.sqrt(J2),J3))
    analytical_r.append(sign(sqrtTwo*np.sqrt(J2),J3))
    
  #Drucker-Prager yeild surface
  yield_zs = np.array([z0,min(analytical_z)])
  yield_rs = r0/z0*((get_yield_surface(uda_path)['PEAKI1']/sqrtThree)-yield_zs)
  yield_ps = yield_zs*(sqrtThree/3.0)
  yield_qs = yield_rs*np.sqrt(threeHalves)

  ###PLOTTING
  ##Plot a
  
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75)
    material_dict = get_yield_surface(uda_path)  
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  else:
    tmp = plt.rcParams['axes.labelsize']
    plt.rcParams['axes.labelsize']='x-small'
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)     
  plt.plot(analytical_z,analytical_r,'-g',linewidth=lineWidth+1,label='Analytical')
  plt.plot(yield_zs,yield_rs,'--k',linewidth=lineWidth+2,label='Yield surface')
  plt.plot(yield_zs,-yield_rs,'--k',linewidth=lineWidth+2)  
  plt.plot(zs,rs,'-r',label='Uintah')  
  plt.xlabel(str_to_mathbf('Isomorphic pressure, z (Pa)'))    
  plt.ylabel(str_to_mathbf('Lode radius, r (Pa)'))
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)    
  ax1.set_xlim(-500,100)
  ax1.set_ylim(-250,250)
  plt.grid(True)    
  plt.legend()
  if BIG_FIGURE:
    plt.title('AreniscaTest 02:\nVertex Treatment (plot a)')
    saveIMG(save_path+'/Test02_verificationPlot_a','1280x960')
  else:
    saveIMG(save_path+'/Test02_verificationPlot_a','640x480')
    plt.rcParams['axes.labelsize']=tmp
  
  ##Plot b
  plt.figure(2)
  plt.clf()
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75)
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)     
  endT = max(times)  
  #Sigma zz
  ax3 = plt.subplot(313)
  plt.plot(analytical_times,analytical_S33,'-g',linewidth=lineWidth+2)
  plt.plot(times,np.array(Szz),'-r')  
  #Add Yield Surface
  #Add Analytical
  plt.xlabel(str_to_mathbf('Time (s)'))  
  plt.ylabel(str_to_mathbf('\sigma_{zz} (Pa)')) 
  ax3.xaxis.set_major_formatter(formatter_int)
  ax3.yaxis.set_major_formatter(formatter_int)  
  ax3.set_xlim(0,endT)
  ax3.set_ylim(-300,100)
  ax3.set_yticks([-300,-200,-100,0,100])
  plt.grid(True)
  #Sigma xx
  ax1 = plt.subplot(311,sharex=ax3)
  plt.plot(analytical_times,analytical_S11,'-g',linewidth=lineWidth+2,label='Analytical')
  plt.plot(times,np.array(Sxx),'-r',label='Uintah')  
  #Add Yield Surface
  #Add Analytical  
  plt.legend(loc=9)
  plt.setp(ax1.get_xticklabels(), visible=False)
  plt.ylabel(str_to_mathbf('\sigma_{xx} (Pa)'))  
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)    
  ax1.set_xlim(0,endT)
  ax1.set_ylim(-400,100)
  ax1.set_yticks([-400,-300,-200,-100,0,100])
  plt.grid(True)
  #Sigma yy
  ax2 = plt.subplot(312,sharex=ax3)
  plt.plot(analytical_times,analytical_S22,'-g',linewidth=lineWidth+2)
  plt.plot(times,np.array(Syy),'-r')  
  #Add Yield Surface
  #Add Analytical 
  plt.setp(ax2.get_xticklabels(), visible=False)
  plt.ylabel(str_to_mathbf('\sigma_{yy} (Pa)'))  
  ax2.yaxis.set_major_formatter(formatter_int)
  ax2.xaxis.set_major_formatter(formatter_int)
  ax2.set_xlim(0,endT)
  ax2.set_ylim(-300,100)
  ax2.set_yticks([-300,-200,-100,0,100])  
  plt.grid(True)
  if BIG_FIGURE:
    plt.title('AreniscaTest 02:\nVertex Treatment (plot b)')
    saveIMG(save_path+'/Test02_verificationPlot_b','1280x960')   
  else:
    saveIMG(save_path+'/Test02_verificationPlot_b','640x480')  
  if SHOW_ON_MAKE:
    plt.show()
  
def test03_postProc(uda_path,save_path,**kwargs):  
  #ABC switches the displayed initial yield surface
  if 'a' in kwargs.keys():
    ABC = 'a'
    partial_string = '(a) - Uniaxial Strain Without Hardening'
  elif 'b' in kwargs.keys():
    ABC = 'b'
    partial_string = '(b) - Uniaxial Strain With Isotropic Hardening'
  elif 'c' in kwargs.keys():
    ABC = 'c'
    partial_string = '(c) - Uniaxial Strain With Kinematic Hardening'
  
  #Extract stress history
  print "Post Processing Test: 03 "+partial_string
  times,sigmas = get_pStress(uda_path)
  ps,qs = get_ps_and_qs(sigmas)
  rs = qs*np.sqrt(2.0/3.0)
  zs = (ps*3.0)/np.sqrt(3.0)  
  material_dict = get_yield_surface(uda_path)
  PEAKI1 = material_dict['PEAKI1']
  J2Yield = J2_at_Yield(uda_path)
  q_yield = np.sqrt(3.0*J2Yield)
  r_yield = np.sqrt(2.0*J2Yield)
  
  ###PLOTTING 
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75)
    material_dict = get_yield_surface(uda_path)  
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)     
    
  plt.plot(np.array(zs),np.array(rs),'-r',label='Uintah')  
  plt.axhline(r_yield,ls='--',color='k',linewidth=lineWidth+1,label='Initial yield surface')
  plt.axhline(-r_yield,ls='--',color='k',linewidth=lineWidth+1)
  plt.xlabel(str_to_mathbf('Isomorphic pressure, z (Pa)'))    
  plt.ylabel(str_to_mathbf('Lode radius, r (Pa)'))   
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)
  ax1.set_ylim(-100,100)
  plt.grid(True)    
  plt.legend()
  
  if BIG_FIGURE:
    plt.title('AreniscaTest 03:\n'+partial_string)
    saveIMG(save_path+'/Test03_verificationPlot_'+ABC,'1280x960')
  else:
    saveIMG(save_path+'/Test03_verificationPlot_'+ABC,'640x480')
  if SHOW_ON_MAKE:
    plt.show()

def test04_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 04 - Curved Yield Surface"
  times,sigmas = get_pStress(uda_path)
  ps,qs = get_ps_and_qs(sigmas)
  rs = qs*np.sqrt(2.0/3.0)
  zs = (ps*3.0)/np.sqrt(3.0)

  ###PLOTTING
  ##Plot a
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75)
    material_dict = get_yield_surface(uda_path)  
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)     

  plt.plot(zs,rs,'-r',label='Uintah')  
  plt.ylabel(str_to_mathbf('Lode radius, r (Pa)'))
  plt.xlabel(str_to_mathbf('Isomorphic pressure, z (Pa)'))  
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)    
  #ax1.set_xlim(-700,300)
  #ax1.set_ylim(-200,200)
  plt.grid(True)  
  
  plot_yield_surface(uda_path,'r_vs_z')
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)   
  #Add Analytical
  plt.legend()
  if BIG_FIGURE:
    plt.title('AreniscaTest 04:\nCurved Yield Surface')
    saveIMG(save_path+'/Test04_verificationPlot','1280x960')
  else:
    saveIMG(save_path+'/Test04_verificationPlot','640x480')
  if SHOW_ON_MAKE:
    plt.show()
  
def test05_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 05 - Hydrostatic Compression Fixed Cap"
  times,sigmas = get_pStress(uda_path)
  ps,qs = get_ps_and_qs(sigmas)
  rs = qs*np.sqrt(2.0/3.0)
  zs = (ps*3.0)/np.sqrt(3.0)
  
  ###PLOTTING
  ##Plot a
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75)
    material_dict = get_yield_surface(uda_path)  
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)     
  
  plt.plot(zs,rs,'-r',label='Uintah')  
  plt.ylabel(str_to_mathbf('Lode radius, r (Pa)'))
  plt.xlabel(str_to_mathbf('Isomorphic pressure, z (Pa)'))  
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)    
  #ax1.set_xlim(-700,300)
  #ax1.set_ylim(-200,200)
  plt.grid(True)  
  
  plot_yield_surface(uda_path,'r_vs_z')
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int) 
  #Add Analytical
  plt.legend()
  if BIG_FIGURE:
    plt.title('AreniscaTest 05:\nHydrostatic Compression Fixed Cap')
    saveIMG(save_path+'/Test05_verificationPlot','1280x960')
  else:
    saveIMG(save_path+'/Test05_verificationPlot','640x480')
  if SHOW_ON_MAKE:
    plt.show()
  
def test06_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 06 - Uniaxial Strain Cap Evolution"
  times,sigmas = get_pStress(uda_path)
  ps,qs = get_ps_and_qs(sigmas)
  rs = qs*np.sqrt(2.0/3.0)
  zs = (ps*3.0)/np.sqrt(3.0)
  
  ###PLOTTING
  ##Plot a
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75)
    material_dict = get_yield_surface(uda_path)  
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)     
  
  plt.plot(zs,rs,'-r',label='Uintah')  
  plt.ylabel(str_to_mathbf('Lode radius, r (Pa)'))
  plt.xlabel(str_to_mathbf('Isomorphic pressure, z (Pa)'))  
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)    
  #ax1.set_xlim(-800,300)
  #ax1.set_ylim(-200,200)
  plt.grid(True)   
  
  plot_yield_surface(uda_path,'r_vs_z')
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int) 
  #Add Analytical
  plt.legend()
  if BIG_FIGURE:
    plt.title('AreniscaTest 06:\nUniaxial Strain Cap Evolution')
    saveIMG(save_path+'/Test06_verificationPlot','1280x960')
  else:
    saveIMG(save_path+'/Test06_verificationPlot','640x480')
  if SHOW_ON_MAKE:
    plt.show()

def test07_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 07 - Hydrostatic Compression with Fixed Cap"
  times,sigmas = get_pStress(uda_path)
  I1s = []
  for sigma in sigmas:
    I1s.append(sigma_I1(sigma))
  times,plasticStrainVol = get_pPlasticStrainVol(uda_path)
  material_dict = get_yield_surface(uda_path)
  P3 = material_dict['P3']
  
  porositys = []
  for I1 in I1s:
    porositys.append(get_porosity(I1,material_dict))

  porositys = 1-np.exp(-(P3+np.array(plasticStrainVol)))

  ###PLOTTING
  ##Plot a
  I1lims = (-8000,0)  
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75) 
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)     
  ax1=eqShear_vs_meanStress(I1s,porositys,I1lims,(0,0.6))
  plt.ylabel(str_to_mathbf('Porosity'))
  plt.xlabel(str_to_mathbf('I_{1}:first invariant of stress tensor (Pa)'))
  plot_crush_curve(uda_path,I1lims)
  ax1.set_ylim(0,0.6)
  ax1.set_xlim(-8100,100)  
  ax1.set_xticks([-8000,-6000,-4000,-2000,0])
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)   
  plt.legend(loc=2)
  if BIG_FIGURE:
    plt.title('AreniscaTest 07:\nHydrostatic Compression with Fixed Cap')
    saveIMG(save_path+'/Test07_verificationPlot','1280x960')
  else:
    saveIMG(save_path+'/Test07_verificationPlot','640x480')
  if SHOW_ON_MAKE:
    plt.show()

def test08_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 08 - Loading/Unloading"
  times,sigmas = get_pStress(uda_path)
  I1s = []
  ps = []
  for sigma in sigmas:
    I1s.append(sigma_I1(sigma))
    ps.append(sigma_I1(sigma)/3.0)
  times,plasticStrainVol = get_pPlasticStrainVol(uda_path)
  times,elasticStrainVol = get_pElasticStrainVol(uda_path)
  totalStrainVol = np.array(elasticStrainVol)+np.array(plasticStrainVol)
  material_dict = get_yield_surface(uda_path)
  P3 = material_dict['P3']

  porositys = 1-np.exp(-(P3+np.array(plasticStrainVol)))

  ###PLOTTING
  ##Plot a
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75,left=0.15)  
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)     
  ax1=eqShear_vs_meanStress(times,-np.array(ps)) 
  
  plt.ylabel(str_to_mathbf('Pressure (Pa)'))
  plt.xlabel(str_to_mathbf('Time (s)'))
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int)
  ax1.tick_params(axis='both',labelsize='small')
  if BIG_FIGURE:
    plt.title('AreniscaTest 08:\nLoading/Unloading (plot a)')
    saveIMG(save_path+'/Test08_verificationPlot_a','1280x960')
  else:
    saveIMG(save_path+'/Test08_verificationPlot_a','640x480')  
  
  ##Plot b
  plt.figure(2)
  plt.clf()
  ax2 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75,left=0.15)  
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')  
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)     
  ax1=eqShear_vs_meanStress(times,totalStrainVol,(0,3.5),(-0.8,0.8))  
  plt.ylabel(str_to_mathbf('Total Volumetric Strain, \epsilon_{v}'))
  plt.xlabel(str_to_mathbf('Time (s)'))
  ax2.xaxis.set_major_formatter(formatter_int)
  ax2.yaxis.set_major_formatter(formatter_int)  
  ax2.tick_params(axis='both',labelsize='small')
  if BIG_FIGURE:
    plt.title('AreniscaTest 08:\nLoading/Unloading (plot b)') 
    saveIMG(save_path+'/Test08_verificationPlot_b','1280x960')
  else:
    saveIMG(save_path+'/Test08_verificationPlot_b','640x480')
    
  ##Plot c
  I1lims = (-7000,0)  
  plt.figure(3)
  plt.clf()
  ax3 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75,left=0.15) 
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')  
  else:
    plt.subplots_adjust(left=0.13,top=0.96,bottom=0.15,right=0.94)     
  eqShear_vs_meanStress(I1s,porositys)  
  plt.ylabel(str_to_mathbf('Porosity'))
  plt.xlabel(str_to_mathbf('I_{1}:first invariant of stress tensor (Pa)'))
  plot_crush_curve(uda_path,I1lims)
  ax3.set_ylim(0,0.8)
  ax3.set_xticks([-6000,-4000,-2000,0])   
  ax3.xaxis.set_major_formatter(formatter_int)
  ax3.yaxis.set_major_formatter(formatter_int)
  ax3.tick_params(axis='both',labelsize='small')
  plt.legend(loc=2)
  
  if True:
    plt.annotate('1',(0,0.39),(0,0.39))
    plt.annotate('2',(-2100,0.39),(-2100,0.39))
    plt.annotate('3',(-2800,0.26),(-2800,0.26))
    plt.annotate('4',(max(I1s),0.235),(max(I1s),0.235))
    plt.annotate('5',(max(I1s),0.725),(max(I1s),0.725))
    plt.annotate('6',(-950,0.725),(-950,0.725))
    plt.annotate('7',(min(I1s),0.06),(min(I1s),0.06))
  
  if BIG_FIGURE:
    plt.title('AreniscaTest 08:\nLoading/Unloading (plot c)')
    saveIMG(save_path+'/Test08_verificationPlot_c','1280x960')
  else:
    saveIMG(save_path+'/Test08_verificationPlot_c','640x480')
  if SHOW_ON_MAKE:
    plt.show()   
  
def test09_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 09 - Fluid Filled Pore Space"
  times,sigmas = get_pStress(uda_path)
  I1s = []
  for sigma in sigmas:
    I1s.append(sigma_I1(sigma))
  times,plasticStrainVol = get_pPlasticStrainVol(uda_path)
  material_dict = get_yield_surface(uda_path)
  P3 = material_dict['P3']
  porositys = 1-np.exp(-(P3+np.array(plasticStrainVol)))
  
  
  ###PLOTTING
  ##Plot a
  I1lims = (-8000,0)  
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  if BIG_FIGURE:
    plt.subplots_adjust(right=0.75) 
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')  
  else:
    plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)
  plt.hold(True)
  plot_crush_curve(uda_path,I1lims)
  ax1=eqShear_vs_meanStress(I1s,porositys)#,I1lims,(0,0.6))  
  plt.ylabel(str_to_mathbf('Porosity'))
  plt.xlabel(str_to_mathbf('I_{1}:first invariant of stress tensor (Pa)'))
  
  
  ax1.xaxis.set_major_formatter(formatter_int)
  ax1.yaxis.set_major_formatter(formatter_int) 
  plt.legend()
  if BIG_FIGURE:
    plt.title('AreniscaTest 09:\nFluid Filled Pore Space')
    saveIMG(save_path+'/Test09_verificationPlot','1280x960')
  else:
    #ax1.set_xticks([-8000,-6000,-4000,-2000,0]) 
    saveIMG(save_path+'/Test09_verificationPlot','640x480')
    
  if SHOW_ON_MAKE:
    plt.show()  
  
def test10_postProc(uda_path,save_path,**kwargs):
  if 'WORKING_PATH' in kwargs:
    working_dir = kwargs['WORKING_PATH']
  
    #Extract stress history
    print "Post Processing Test: 10 - Transient Stress Eigenvalues with Constant Eigenvectors"
    times,sigmas = get_pStress(uda_path)
    Sxx = []
    Syy = []
    Szz = []
    for sigma in sigmas:
      Sxx.append(sigma[0][0])
      Syy.append(sigma[1][1])
      Szz.append(sigma[2][2])  
      
    #Analytical solution
    material_dict = get_yield_surface(uda_path)
    def_times,Fs = get_defTable(uda_path,working_dir)
    tau_yield = material_dict['PEAKI1']/1e10
    bulk_mod = material_dict['B0']
    shear_mod = material_dict['G0']
    
    analytical_times,analytical_sigmas,epsils=defTable_to_J2Solution(def_times,Fs,bulk_mod,shear_mod,tau_yield,num_substeps=10)
   
    analytical_Sxx = []
    analytical_Syy = []
    analytical_Szz = []
    for sigma in analytical_sigmas:
      analytical_Sxx.append(sigma[0][0])
      analytical_Syy.append(sigma[1][1])
      analytical_Szz.append(sigma[2][2])

    ###PLOTTING
    plt.figure(1)
    plt.clf()
    ax1 = plt.subplot(111)
    if BIG_FIGURE:
      plt.subplots_adjust(right=0.75)
      param_text = material_dict['material string']
      plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
    else:
      plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)       
   
    #analytical solution
    plt.plot(analytical_times,np.array(analytical_Sxx)/1e6,':r',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{xx}'))  
    plt.plot(analytical_times,np.array(analytical_Syy)/1e6,'--g',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{yy}'))     
    plt.plot(analytical_times,np.array(analytical_Szz)/1e6,'-.b',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{zz}'))    
    #simulation results
    plt.plot(times,np.array(Sxx)/1e6,'-r',label=str_to_mathbf('Uintah \sigma_{xx}'))       
    plt.plot(times,np.array(Syy)/1e6,'-g',label=str_to_mathbf('Uintah \sigma_{yy}'))   
    plt.plot(times,np.array(Szz)/1e6,'-b',label=str_to_mathbf('Uintah \sigma_{zz}'))      
    
    ax1.set_xlim(0,2.25)   
    ax1.xaxis.set_major_formatter(formatter_int)
    ax1.yaxis.set_major_formatter(formatter_int)     
    #labels
    plt.grid(True)         
    plt.xlabel(str_to_mathbf('Time (s)'))
    plt.ylabel(str_to_mathbf('Stress (MPa)'))
    if BIG_FIGURE:
      plt.legend(loc='upper right', bbox_to_anchor=(1.38,1.12))
      plt.title('AreniscaTest 10:\nTransient Stress Eigenvalues with Constant Eigenvectors') 
      saveIMG(save_path+'/Test10_verificationPlot','1280x960')
    else:
      tmp = plt.rcParams['legend.fontsize']
      plt.rcParams['legend.fontsize']='x-small'
      plt.legend(loc=7)
      saveIMG(save_path+'/Test10_verificationPlot','640x480')
      plt.rcParams['legend.fontsize']=tmp
    if SHOW_ON_MAKE:
      plt.show()
  
  else:
    print '\nERROR: need working directory to post process this problem'
  
def test11_postProc(uda_path,save_path,**kwargs):
  if 'WORKING_PATH' in kwargs:
    working_dir = kwargs['WORKING_PATH']
  
    #Extract stress and strain history
    print "Post Processing Test: 11 - Uniaxial Strain J2 Plasticity"
    times,sigmas = get_pStress(uda_path)
    times,epsils = get_epsilons(uda_path)
    exx = []
    eyy = []
    ezz = []
    for epsil in epsils:
      exx.append(epsil[0][0])
      eyy.append(epsil[1][1])
      ezz.append(epsil[2][2])  
    Sxx = []
    Syy = []
    Szz = []
    for sigma in sigmas:
      Sxx.append(sigma[0][0])
      Syy.append(sigma[1][1])
      Szz.append(sigma[2][2])      
      
    #Analytical solution
    material_dict = get_yield_surface(uda_path)
    def_times,Fs = get_defTable(uda_path,working_dir)
    tau_yield = material_dict['PEAKI1']*material_dict['FSLOPE']    
    bulk_mod = material_dict['B0']
    shear_mod = material_dict['G0']
    
    analytical_times,analytical_sigmas,epsils2=defTable_to_J2Solution(def_times,Fs,bulk_mod,shear_mod,tau_yield,num_substeps=1000)
   
    analytical_Sxx = []
    analytical_Syy = []
    analytical_Szz = []
    for sigma in analytical_sigmas:
      analytical_Sxx.append(sigma[0][0])
      analytical_Syy.append(sigma[1][1])
      analytical_Szz.append(sigma[2][2])
   
    analytical_e11 = []
    analytical_e22 = []
    analytical_e33 = []
    for epsil in epsils2:
      analytical_e11.append(epsil[0][0])
      analytical_e22.append(epsil[1][1])
      analytical_e33.append(epsil[2][2])
   

    ###PLOTTING
    plt.figure(1)
    plt.clf()
    ax1 = plt.subplot(111)
    if BIG_FIGURE:
      plt.subplots_adjust(right=0.75)
      param_text = material_dict['material string']
      plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
    else:
      plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)       
      
    ax1.xaxis.set_major_formatter(formatter_int)
    ax1.yaxis.set_major_formatter(formatter_int)    
    plt.plot(np.array(analytical_e11),np.array(analytical_Sxx)/1e6,'--g',linewidth=lineWidth+1,label=str_to_mathbf('Analytical'))    
    plt.plot(np.array(exx),np.array(Sxx)/1e6,'-r',label=str_to_mathbf('Uintah'))
    plt.xlabel(str_to_mathbf('\epsilon_{A}'))
    plt.ylabel(str_to_mathbf('\sigma_{A} (MPa)'))     
    plt.legend()
    plt.grid(True)
    if BIG_FIGURE:
      plt.title('AreniscaTest 11:\nUniaxial Strain J2 Plasticity (plot a)')
      saveIMG(save_path+'/Test11_verificationPlot_a','1280x960')      
    else:
      saveIMG(save_path+'/Test11_verificationPlot_a','640x480')
    
    plt.figure(2)
    plt.clf()
    ax2 = plt.subplot(111)
    if BIG_FIGURE:
      plt.subplots_adjust(right=0.75)
      param_text = material_dict['material string']
      plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
    else:
      plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96) 
    ax2.xaxis.set_major_formatter(formatter_int)
    ax2.yaxis.set_major_formatter(formatter_int)        
    plt.plot(np.array(analytical_e11),np.array(analytical_Syy)/1e6,'--g',linewidth=lineWidth+1,label=str_to_mathbf('Analytical'))
    plt.plot(np.array(exx),np.array(Syy)/1e6,'-r',label=str_to_mathbf('Uintah'))
    plt.xlabel(str_to_mathbf('\epsilon_{A}'))
    plt.ylabel(str_to_mathbf('\sigma_{L} (MPa)')) 
    plt.legend()
    plt.grid(True)
    if BIG_FIGURE:
      plt.title('AreniscaTest 11:\nUniaxial Strain J2 Plasticity (plot b)') 
      saveIMG(save_path+'/Test11_verificationPlot_b','1280x960')      
    else:
      saveIMG(save_path+'/Test11_verificationPlot_b','640x480') 
      
  
  else:
    print '\nERROR: need working directory to post process this problem'  
  
def test12_postProc(uda_path,save_path,**kwargs):
  if 'WORKING_PATH' in kwargs:
    working_dir = kwargs['WORKING_PATH']
  
    #Extract stress history
    print "Post Processing Test: 12 - Pure Isochoric Strain Rates in Different Directions"
    times,sigmas = get_pStress(uda_path)
    Sxx = []
    Syy = []
    Szz = []
    Sxy = []
    for sigma in sigmas:
      Sxx.append(sigma[0][0])
      Syy.append(sigma[1][1])
      Szz.append(sigma[2][2])
      Sxy.append(sigma[0][1])
      
    #Analytical solution
    material_dict = get_yield_surface(uda_path)
    def_times,Fs = get_defTable(uda_path,working_dir)
    tau_yield = material_dict['PEAKI1']/1e10
    bulk_mod = material_dict['B0']
    shear_mod = material_dict['G0']
    
    analytical_times,analytical_sigmas,epsils=defTable_to_J2Solution(def_times,Fs,bulk_mod,shear_mod,tau_yield,num_substeps=10)
   
    analytical_Sxx = []
    analytical_Syy = []
    analytical_Szz = []
    analytical_Sxy = []
    for sigma in analytical_sigmas:
      analytical_Sxx.append(sigma[0][0])
      analytical_Syy.append(sigma[1][1])
      analytical_Szz.append(sigma[2][2])
      analytical_Sxy.append(sigma[0][1])

    ###PLOTTING
    plt.figure(1)
    plt.clf()
    ax1 = plt.subplot(111)
    if BIG_FIGURE:
      plt.subplots_adjust(right=0.75)
      param_text = material_dict['material string']
      plt.figtext(0.77,0.68,param_text,ha='left',va='top',size='x-small')
    else:
      plt.subplots_adjust(left=0.15,top=0.96,bottom=0.15,right=0.96)       
   
    #analytical solution
    plt.plot(analytical_times,np.array(analytical_Sxx)/1e6,':r',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{xx}'))  
    plt.plot(analytical_times,np.array(analytical_Syy)/1e6,':g',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{yy}'))     
    plt.plot(analytical_times,np.array(analytical_Szz)/1e6,':b',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{zz}'))
    plt.plot(analytical_times,np.array(analytical_Sxy)/1e6,':k',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{xy}'))
    #simulation results
    plt.plot(times,np.array(Sxx)/1e6,'-r',label=str_to_mathbf('Uintah \sigma_{xx}'))       
    plt.plot(times,np.array(Syy)/1e6,'-g',label=str_to_mathbf('Uintah \sigma_{yy}'))   
    plt.plot(times,np.array(Szz)/1e6,'-b',label=str_to_mathbf('Uintah \sigma_{zz}'))
    plt.plot(times,np.array(Sxy)/1e6,'-k',label=str_to_mathbf('Uintah \sigma_{xy}'))
    
    ax1.set_xlim(0,5.25)
    #ax1.set_ylim(0,3) 
    ax1.xaxis.set_major_formatter(formatter_int)
    ax1.yaxis.set_major_formatter(formatter_int)     
    #labels
    plt.grid(True)         
    plt.xlabel(str_to_mathbf('Time (s)'))
    plt.ylabel(str_to_mathbf('Stress (MPa)'))
    if BIG_FIGURE:
      plt.legend(loc='upper right', bbox_to_anchor=(1.38,1.12))
      plt.title('AreniscaTest 12:\nPure Isochoric Strain Rates in Different Directions') 
      saveIMG(save_path+'/Test12_verificationPlot','1280x960')
    else:
      tmp = plt.rcParams['legend.fontsize']
      plt.rcParams['legend.fontsize']='x-small'
      plt.legend(loc=7)
      saveIMG(save_path+'/Test12_verificationPlot','640x480')
      plt.rcParams['legend.fontsize']=tmp
    if SHOW_ON_MAKE:
      plt.show()

  else:
    print '\nERROR: need working directory to post process this problem'  

def test13_postProc(uda_path,save_path,**kwargs):
  COLORS = ['Black','Blue','Magenta','Red','Green']
  if 'WORKING_PATH' in kwargs:
    working_dir = kwargs['WORKING_PATH']  


    #Plot Constants
    Xlims = (-450,50)
    Ylims = (-100,100)  
    formatter = ticker.FormatStrFormatter('$\mathbf{%g}$')  
    plt.figure(1)
    plt.hold(True)
    plt.clf()

    material_dict = get_yield_surface(uda_path)    
    PEAKI1 = material_dict['PEAKI1']
    FSLOPE = material_dict['FSLOPE']
    T1 = material_dict['T1']
    T2 = material_dict['T2']    

    def_times,Fs = get_defTable(uda_path,working_dir)
    A = Fs[1][0][0]
    #As = Fs[10][0][0]
    K = material_dict['B0']
    G = material_dict['G0']
    C = K+(4.0/3.0)*G
    Y = FSLOPE*PEAKI1*1.732
    YS = FSLOPE*PEAKI1
    
    #uniaxial strain (unscaled)
    analytical_exx = [0.0,
		      (Y/(2.0*G)),
		      np.log(A),
		      ]  
    
    analytical_Sxx=[0.0,
		(C*Y)/(2.0*G),
		((C-K)*Y)/(2*G)+K*np.log(A),
		]      
    
    #uniaxial strain (scaled)
    #analytical_exx = np.array([0.0,
			      #(Y/(2.0*G)),
			      #np.log(A),
			      #np.log(A)-(Y)/(G),
			      #0.0
			      #])/(Y/(2.0*G))

    #analytical_Sxx = np.array([0.0,
			      #(C*Y)/(2.0*G),
			      #((C-K)*Y)/(2*G)+K*np.log(A),
			      #K*np.log(A)-((C+K)*Y)/(2*G),
			      #(K-C)*Y/(2*G)
			      #])/((C*Y)/(2.0*G))
    
    #pure shear (unscaled)
    #analytical_exx = np.array([0.0,
	#		      (YS/(2.0*G)),
	#		      np.log(As),
	#		      ])
    #analytical_Sxx = np.array([0.0,
	#		      (YS),
	#		      (YS),
	#		      ])


    #Extract stress history
    print "Post Processing Test: 13 "
    times,sigmas = get_pStress(uda_path)
    times,epsils = get_epsilons(uda_path)
    exx = []
    eyy = []
    ezz = []
    exy = []
    for epsil in epsils:
      exx.append(epsil[0][0])
      eyy.append(epsil[1][1])
      ezz.append(epsil[2][2])
      exy.append(epsil[0][1])
    Sxx = []
    Syy = []
    Szz = []
    Sxy = []
    for sigma in sigmas:
      Sxx.append(sigma[0][0])
      Syy.append(sigma[1][1])
      Szz.append(sigma[2][2])
      Sxy.append(sigma[0][1])
      
    scaled_exx = ((2.0*G)/Y)*np.array(exx)
    scaled_Sxx = ((2.0*G)/(C*Y))*np.array(Sxx)
    scaled_Syy = ((2.0*G)/(C*Y))*np.array(Syy)
    #S = np.array(Sxx) - np.array(Syy)
    S = np.array(Sxx)
    #E = np.array(exy)
    ###PLOTTING

    ax1 = plt.subplot(111)
    plt.subplots_adjust(right=0.75)
    #param_text = material_dict['material string']
    #plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')   
    eqShear_vs_meanStress(exx,S,LINE_LABEL = 'T1='+format(T1,'1.3e')+' T2='+format(T2,'1.3e'))
    #eqShear_vs_meanStress(E,S,LINE_LABEL = 'T1='+format(T1,'1.3e')+' T2='+format(T2,'1.3e'),COLOR=COLORS[idx])
    plt.plot(analytical_exx,analytical_Sxx,'--',color='Red',label='Analytical solution for rate independent case.')
    plt.title('AreniscaTest 13:')
    plt.ylabel(str_to_mathbf('\sigma_{xx}'))
    plt.xlabel(str_to_mathbf('\epsilon_{xx}'))  
    #plt.ylabel(str_to_mathbf('\sigma_{xy}'))
    #plt.xlabel(str_to_mathbf('\epsilon_{xy}'))
    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)   
    plt.legend()
    saveIMG(save_path+'/Test13_verificationPlot','1280x960')
    if SHOW_ON_MAKE:
      plt.show()   
  
  else:
    print '\nERROR: need working directory to post process this problem'
