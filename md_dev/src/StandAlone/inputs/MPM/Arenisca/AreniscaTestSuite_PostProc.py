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

SHOW_ON_MAKE = FALSE

#Useful constants
sqrtThree = np.sqrt(3.0)
twoThirds = 2.0/3.0
threeHalves = 3.0/2.0

#Set matplotlib defaults to desired values
#Set the legend to best fit
fontSize = 16

markers = None
plt.rcParams['legend.loc']='best'
#Set font size
plt.rcParams['mathtext.it'] = 'serif:bold'
plt.rcParams['mathtext.rm'] = 'serif:bold'
plt.rcParams['mathtext.sf'] = 'serif:bold'
plt.rcParams['font.size']=fontSize
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelsize']='medium'
#plt.rcParams['axes.labelweight']='bold'
plt.rcParams['legend.fontsize']='medium'
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
 

def savePNG(name,size='1920x1080'):
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
  plt.savefig(name+'.png', bbox_inches=0, dpi=plt.rcParams['figure.dpi']) 

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
  return ps,qs
  
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
  ups_file = uda_path+'/input.xml'
  F = open(ups_file,'r')
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
    return r'$\mathbf{'+lead+r'\cdot{}10^{'+exp+'}}$' 

def eqShear_vs_meanStress(xs,ys,Xlims=False,Ylims=False,LINE_LABEL='Uintah',GRID=True):

  ax1 = plt.subplot(111)
  plt.plot(np.array(xs),np.array(ys),'-r',label=LINE_LABEL)
  
  plt.xlabel(str_to_mathbf('Mean Stress, p (Pa)'))
  plt.ylabel(str_to_mathbf('Equivalent Shear Stress, q, (Pa)'))
  
  formatter_int = ticker.FormatStrFormatter('$\mathbf{%g}$')
  formatter_exp = ticker.FuncFormatter(exp_fmt)
  
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

def plot_crush_curve(uda_path,I1lims=[-10000,0]):
  nPoints = 500
  material_dict = get_yield_surface(uda_path)
  P0 = material_dict['P0']
  P1 = material_dict['P1']
  P3 = material_dict['P3']
  I1s = np.linspace(I1lims[0],I1lims[1],nPoints)
  porosity = P3*np.exp(P1*(I1s-P0))
  plt.plot(I1s,porosity,'--g',linewidth=lineWidth+1,label='Analytical crush curve')

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

def J2VM(epsil_dot,dt,sig_Beg,K,G,tau_y):
  #J2 plasticity Von misses material model for 3D
  #Inputs: epsil_dot, dt, sig_Beg, K, G, tau_y
  #Outputs: epsil_Elastic_dot, epsil_Plastic_dot, sig_End
  #Initialize the trial stress state
  sig_Trial = sig_Beg+((2*G*sigma_dev(epsil_dot))+3*K*sigma_iso(epsil_dot))*dt
  #Determine if this is below, on, or above the yeild surface
  test = sigma_mag(sigma_dev(sig_Trial))/(np.sqrt(2.0)*tau_y)
  if test<=1:
      #Stress state is elastic
      sig_End = sig_Trial
      epsil_Plastic_dot = np.zeros((3,3))
      epsil_Elastic_dot = epsil_dot
  elif test>1:
      #Stress state elastic-plastic
      sig_End = (sigma_dev(sig_Trial)/test)#+sigma_iso(sig_Trial)
      #Evaluate the consistent stress rate
      #sig_dot = (sig_End-sig_Beg)/test
      #Apply hookes law to get the elastic strain rate
      #epsil_Elastic_dot = sigma_dev(sig_dot)/(2*G)# + sigma_iso(sig_dot)/(3*K)
      #Apply strain rate decomposition relationship to get plastic strain rate
      #epsil_Plastic_dot = epsil_dot-epsil_Elastic_dot
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
    epsils.append(np.array([[np.log(sum(F[0])),0,0],[0,np.log(sum(F[1])),0],[0,0,np.log(sum(F[2]))]]))
    
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
  
  kappa_initial = (P0+FSLOPE*CR*PEAKI1)/(1.0+FSLOPE*CR)
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
  
  kappa_initial = (P0+FSLOPE*CR*PEAKI1)/(1.0+FSLOPE*CR)
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
      J2 = ((FSLOPE**2)*((I1-PEAKI1+3.0*Pf0)**2))*(1.0-((I1+CR*FSLOPE*I1-P0-CR*FSLOPE*PEAKI1+3.0*Pf0+3.0*CR*FSLOPE*Pf0)**2/((CR**2)*(FSLOPE**2)*(P0-PEAKI1)**2)))
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
  formatter = ticker.FormatStrFormatter('$\mathbf{%g}$') 
  plt.figure(1)
  plt.clf()
  plt.subplots_adjust(right=0.75)
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
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
  ax2.xaxis.set_major_formatter(formatter)
  ax2.yaxis.set_major_formatter(formatter)
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
  ax1.xaxis.set_major_formatter(formatter)
  ax1.yaxis.set_major_formatter(formatter)
  ax1.set_yticks([0,-20,-40,-60])  
  plt.ylabel(str_to_mathbf('\sigma_{xx} (Pa)')) 
  plt.title('AreniscaTest 01:\nUniaxial Compression With Rotation')
  plt.legend()
  #savePNG(save_path+'/Test01_verificationPlot','1280x960')
  if SHOW_ON_MAKE:
    plt.show()
  
def test02_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 02 - Vertex Treatment"
  times,sigmas = get_pStress(uda_path)
  ps,qs = get_ps_and_qs(sigmas)
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
  analytical_S11 = np.array([0,-850.0/3.0,(-50.0/3.0)*(9.0+4.0*np.sqrt(6.0)),(-50.0/3.0)*(9.0+4.0*np.sqrt(6.0)),(50.0/3.0)*(2.0*np.sqrt(6)-3.0),160.0*np.sqrt(twoThirds)-110.0])
  analytical_S22 = np.array([0,-850.0/3.0,(50.0/3.0)*(2.0*np.sqrt(6.0)-9.0),(50.0/3.0)*(2.0*np.sqrt(6.0)-9.0),(-50.0/3.0)*(3.0+np.sqrt(6.0)),(-10.0/3.0)*(33.0+8.0*np.sqrt(6.0))])
  analytical_S33 = np.array([0,-850.0/3.0,(50.0/3.0)*(2.0*np.sqrt(6.0)-9.0),(50.0/3.0)*(2.0*np.sqrt(6.0)-9.0),(-50.0/3.0)*(3.0+np.sqrt(6.0)),(-10.0/3.0)*(33.0+8.0*np.sqrt(6.0))])
  analytical_mean = (analytical_S11+analytical_S22+analytical_S33)/3.0
  analytical_I1 = analytical_S11+analytical_S22+analytical_S33
  tmp = (1.0/3.0)*analytical_I1
  analytical_s1 = analytical_S11-tmp
  analytical_s2 = analytical_S22-tmp
  analytical_s3 = analytical_S33-tmp
  analytical_J2 = (1.0/2.0)*(pow(analytical_s1,2)+pow(analytical_s2,2)+pow(analytical_s3,2))
  analytical_J3 = (1.0/3.0)*(pow(analytical_s1,3)+pow(analytical_s2,3)+pow(analytical_s3,3))
  analytical_z = analytical_I1/sqrtThree
  analytical_q = []
  for idx,J2 in enumerate(analytical_J2):
    J3 = analytical_J3[idx]
    analytical_q.append(sign(sqrtThree*np.sqrt(J2),J3))
  #Drucker-Prager yeild surface
  yield_zs = np.array([z0,min(analytical_z)])
  yield_rs = r0/z0*((get_yield_surface(uda_path)['PEAKI1']/sqrtThree)-yield_zs)
  yield_ps = yield_zs*(sqrtThree/3.0)
  yield_qs = yield_rs*np.sqrt(threeHalves)

  ###PLOTTING
  formatter = ticker.FormatStrFormatter('$\mathbf{%g}$')
  ##Plot a
  
  
  plt.figure(1)
  plt.clf()
  plt.subplot(111)
  plt.subplots_adjust(right=0.75)
  material_dict = get_yield_surface(uda_path)  
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')  
  plt.plot(analytical_mean,analytical_q,'-g',linewidth=lineWidth+1,label='Analytical')
  plt.plot(yield_ps,yield_qs,'--k',linewidth=lineWidth+2,label='Yield surface')
  plt.plot(yield_ps,-yield_qs,'--k',linewidth=lineWidth+2)
  eqShear_vs_meanStress(ps,qs,(-300,60),(-300,300))  
  plt.title('AreniscaTest 02:\nVertex Treatment (plot a)')  
  plt.legend()
  savePNG(save_path+'/Test02_verificationPlot_a','1280x960')
  
  ##Plot b
  plt.figure(2)
  plt.clf()
  plt.subplots_adjust(right=0.75)
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')   
  endT = max(times)  
  #Sigma zz
  ax3 = plt.subplot(313)
  plt.plot(analytical_times,analytical_S33,'-g',linewidth=lineWidth+2)
  plt.plot(times,np.array(Szz),'-r')  
  #Add Yield Surface
  #Add Analytical
  plt.xlabel(str_to_mathbf('Time (s)'))  
  plt.ylabel(str_to_mathbf('\sigma_{zz} (Pa)')) 
  ax3.yaxis.set_major_formatter(formatter)  
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
  plt.legend()
  plt.setp(ax1.get_xticklabels(), visible=False)
  plt.ylabel(str_to_mathbf('\sigma_{xx} (Pa)'))  
  plt.title('AreniscaTest 02:\nVertex Treatment (plot b)')
  ax1.xaxis.set_major_formatter(formatter)
  ax1.yaxis.set_major_formatter(formatter)    
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
  ax2.yaxis.set_major_formatter(formatter)    
  ax2.set_xlim(0,endT)
  ax2.set_ylim(-300,100)
  ax2.set_yticks([-300,-200,-100,0,100])  
  plt.grid(True)
  savePNG(save_path+'/Test02_verificationPlot_b','1280x960')   
  
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
  material_dict = get_yield_surface(uda_path)
  PEAKI1 = material_dict['PEAKI1']
  J2Yield = J2_at_Yield(uda_path)
  q_yield = np.sqrt(3.0*J2Yield)
  
  #print 'J2Yield : ',J2Yield
  #print 'q_yield : ',q_yield
  
  ###PLOTTING
  Xlims = (-450,50)
  Ylims = (-100,100)  
  formatter = ticker.FormatStrFormatter('$\mathbf{%g}$')  
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  plt.subplots_adjust(right=0.75)
  material_dict = get_yield_surface(uda_path)  
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')   
  eqShear_vs_meanStress(ps,qs,Xlims,Ylims,)
  plt.title('AreniscaTest 03:\n'+partial_string)
  plt.plot(Xlims,(q_yield,q_yield),'--k',linewidth=lineWidth+1,label='Initial yield surface')
  plt.plot(Xlims,(-q_yield,-q_yield),'--k',linewidth=lineWidth+1)
  ax1.xaxis.set_major_formatter(formatter)
  ax1.yaxis.set_major_formatter(formatter)   
  plt.legend()
  savePNG(save_path+'/Test03_verificationPlot_'+ABC,'1280x960')
  if SHOW_ON_MAKE:
    plt.show()

def test04_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 04 - Curved Yield Surface"
  times,sigmas = get_pStress(uda_path)
  ps,qs = get_ps_and_qs(sigmas)

  ###PLOTTING
  formatter = ticker.FormatStrFormatter('$\mathbf{%g}$')
  ##Plot a
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  plt.subplots_adjust(right=0.75)
  material_dict = get_yield_surface(uda_path)  
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')  
  eqShear_vs_meanStress(ps,qs,(-700,300),(-200,200))
  plt.title('AreniscaTest 04:\nCurved Yield Surface')
  plot_yield_surface(uda_path,'q_vs_p')
  ax1.xaxis.set_major_formatter(formatter)
  ax1.yaxis.set_major_formatter(formatter)   
  #Add Analytical
  plt.legend()
  savePNG(save_path+'/Test04_verificationPlot','1280x960')
  if SHOW_ON_MAKE:
    plt.show()
  
def test05_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 05 - Hydrostatic Compression Fixed Cap"
  times,sigmas = get_pStress(uda_path)
  ps,qs = get_ps_and_qs(sigmas)

  ###PLOTTING
  formatter = ticker.FormatStrFormatter('$\mathbf{%g}$')
  ##Plot a
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  plt.subplots_adjust(right=0.75)
  material_dict = get_yield_surface(uda_path)  
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')  
  eqShear_vs_meanStress(ps,qs,(-700,300),(-200,200))
  plt.title('AreniscaTest 05:\nHydrostatic Compression Fixed Cap')
  plot_yield_surface(uda_path,'q_vs_p')
  ax1.xaxis.set_major_formatter(formatter)
  ax1.yaxis.set_major_formatter(formatter) 
  #Add Analytical
  plt.legend()
  savePNG(save_path+'/Test05_verificationPlot','1280x960')
  if SHOW_ON_MAKE:
    plt.show()
  
def test06_postProc(uda_path,save_path,**kwargs):
  #Extract stress history
  print "Post Processing Test: 06 - Uniaxial Strain Cap Evolution"
  times,sigmas = get_pStress(uda_path)
  ps,qs = get_ps_and_qs(sigmas)

  ###PLOTTING
  formatter = ticker.FormatStrFormatter('$\mathbf{%g}$')
  ##Plot a
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  plt.subplots_adjust(right=0.75)
  material_dict = get_yield_surface(uda_path)  
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')    
  eqShear_vs_meanStress(ps,qs,(-800,300),(-200,200))
  plt.title('AreniscaTest 06:\nUniaxial Strain Cap Evolution')
  plot_yield_surface(uda_path,'q_vs_p')
  ax1.xaxis.set_major_formatter(formatter)
  ax1.yaxis.set_major_formatter(formatter) 
  #Add Analytical
  plt.legend()
  savePNG(save_path+'/Test06_verificationPlot','1280x960')
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
  porosity = P3+np.array(plasticStrainVol)
  
  ###PLOTTING
  formatter = ticker.FormatStrFormatter('$\mathbf{%g}$')
  ##Plot a
  I1lims = (-8000,0)  
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  plt.subplots_adjust(right=0.75) 
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  ax1=eqShear_vs_meanStress(I1s,porosity,I1lims,(0,0.6))
  plt.title('AreniscaTest 07:\nHydrostatic Compression with Fixed Cap')
  plt.ylabel(str_to_mathbf('Porosity'))
  plt.xlabel(str_to_mathbf('I_{1}:first invariant of stress tensor (Pa)'))
  plot_crush_curve(uda_path,I1lims)
  #ax1.set_xticks([-9000,-7000,-5000,-3000,-1000,0])
  ax1.set_xticks([-8000,-6000,-4000,-2000,0])
  ax1.xaxis.set_major_formatter(formatter)
  ax1.yaxis.set_major_formatter(formatter)   
  plt.legend()
  savePNG(save_path+'/Test07_verificationPlot','1280x960')
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
  porosity = P3+np.array(plasticStrainVol)

  ###PLOTTING
  int_formatter = ticker.FormatStrFormatter('$\mathbf{%g}$')
  exp_formatter = ticker.FuncFormatter(exp_fmt)
  ##Plot a
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  plt.subplots_adjust(right=0.75,left=0.15)  
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')  
  ax1=eqShear_vs_meanStress(times,-np.array(ps),(0,3.5),(-500,2000))
  plt.title('AreniscaTest 08:\nLoading/Unloading (plot a)')  
  plt.ylabel(str_to_mathbf('Pressure (Pa)'))
  plt.xlabel(str_to_mathbf('Time (s)'))
  ax1.xaxis.set_major_formatter(int_formatter)
  ax1.yaxis.set_major_formatter(exp_formatter)
  ax1.tick_params(axis='both',labelsize='small')
  savePNG(save_path+'/Test08_verificationPlot_a','1280x960')  
  
  ##Plot b
  plt.figure(2)
  plt.clf()
  ax2 = plt.subplot(111)
  plt.subplots_adjust(right=0.75,left=0.15)  
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')  
  ax1=eqShear_vs_meanStress(times,totalStrainVol,(0,3.5),(-0.8,0.8))
  plt.title('AreniscaTest 08:\nLoading/Unloading (plot b)')  
  plt.ylabel(str_to_mathbf('Total Volumetric Strain, \epsilon_{v}'))
  plt.xlabel(str_to_mathbf('Time (s)'))
  ax2.xaxis.set_major_formatter(int_formatter)
  ax2.yaxis.set_major_formatter(int_formatter)  
  ax2.tick_params(axis='both',labelsize='small')
  savePNG(save_path+'/Test08_verificationPlot_b','1280x960')

  ##Plot c
  I1lims = (-7000,0)  
  plt.figure(3)
  plt.clf()
  ax3 = plt.subplot(111)
  plt.subplots_adjust(right=0.75,left=0.15) 
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')  
  eqShear_vs_meanStress(I1s,porosity,I1lims,(0,1.6))
  plt.title('AreniscaTest 08:\nLoading/Unloading (plot c)')
  plt.ylabel(str_to_mathbf('Porosity'))
  plt.xlabel(str_to_mathbf('I_{1}:first invariant of stress tensor (Pa)'))
  plot_crush_curve(uda_path,I1lims)
  #ax1.set_xticks([-9000,-7000,-5000,-3000,-1000,0])
  ax3.set_xticks([-7000,-5000,-3000,-1000,1000])
  ax3.xaxis.set_major_formatter(exp_formatter)
  ax3.yaxis.set_major_formatter(int_formatter)
  ax3.tick_params(axis='both',labelsize='small')
  plt.legend()
  savePNG(save_path+'/Test08_verificationPlot_c','1280x960')
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
  porosity = P3+np.array(plasticStrainVol)
  
  
  ###PLOTTING
  formatter = ticker.FormatStrFormatter('$\mathbf{%g}$')
  ##Plot a
  I1lims = (-8000,0)  
  plt.figure(1)
  plt.clf()
  ax1 = plt.subplot(111)
  plt.subplots_adjust(right=0.75) 
  param_text = material_dict['material string']
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')   
  ax1=eqShear_vs_meanStress(I1s,porosity)#,I1lims,(0,0.6))
  plt.title('AreniscaTest 09:\nFluid Filled Pore Space')
  plt.ylabel(str_to_mathbf('Porosity'))
  plt.xlabel(str_to_mathbf('I_{1}:first invariant of stress tensor (Pa)'))
  plot_crush_curve(uda_path,I1lims)
  #ax1.set_xticks([-8000,-6000,-4000,-2000,0]) 
  ax1.xaxis.set_major_formatter(formatter)
  ax1.yaxis.set_major_formatter(formatter) 
  plt.legend()
  savePNG(save_path+'/Test09_verificationPlot','1280x960')
  if SHOW_ON_MAKE:
    plt.show()  
  
def test10_postProc(uda_path,save_path,**kwargs):
  if 'WORKING_PATH' in kwargs:
    working_dir = kwargs['WORKING_PATH']
  
    #Extract stress history
    print "Post Processing Test: 10 - Pure Isochoric Strain Rates"
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
    
    analytical_times,analytical_sigmas,epsils=defTable_to_J2Solution(def_times,Fs,bulk_mod,shear_mod,tau_yield,num_substeps=1000)
   
    analytical_Sxx = []
    analytical_Syy = []
    analytical_Szz = []
    for sigma in analytical_sigmas:
      analytical_Sxx.append(sigma[0][0])
      analytical_Syy.append(sigma[1][1])
      analytical_Szz.append(sigma[2][2])

    ###PLOTTING
    formatter = ticker.FormatStrFormatter('$\mathbf{%g}$') 
    plt.figure(1)
    plt.clf()
    ax1 = plt.subplot(111)       
    plt.subplots_adjust(right=0.75)
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
   
    #analytical solution
    plt.plot(analytical_times,np.array(analytical_Sxx)/1e6,':r',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{xx}'))  
    plt.plot(analytical_times,np.array(analytical_Syy)/1e6,'--g',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{yy}'))     
    plt.plot(analytical_times,np.array(analytical_Szz)/1e6,'-.b',linewidth=lineWidth+2,label=str_to_mathbf('Analytical \sigma_{zz}'))    
    #simulation results
    plt.plot(times,np.array(Sxx)/1e6,'-r',label=str_to_mathbf('Uintah \sigma_{xx}'))       
    plt.plot(times,np.array(Syy)/1e6,'-g',label=str_to_mathbf('Uintah \sigma_{yy}'))   
    plt.plot(times,np.array(Szz)/1e6,'-b',label=str_to_mathbf('Uintah \sigma_{zz}'))      
    #labels
    plt.legend(loc='upper right', bbox_to_anchor=(1.38,1.12))
    plt.grid(True)
    plt.title('AreniscaTest 10:\nPurely Isochoric Strain Rates')      
    plt.xlabel(str_to_mathbf('Time (s)'))
    plt.ylabel(str_to_mathbf('Stress (Mpa)'))
    savePNG(save_path+'/Test10_verificationPlot','1280x960')
    
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
    tau_yield = material_dict['PEAKI1']/1e10
    bulk_mod = material_dict['B0']
    shear_mod = material_dict['G0']
    
    analytical_times,analytical_sigmas,epsils=defTable_to_J2Solution(def_times,Fs,bulk_mod,shear_mod,tau_yield,num_substeps=1000)
   
    analytical_e11 = []
    analytical_e22 = []
    analytical_e33 = []
    for epsil in epsils:
      analytical_e11.append(epsil[0][0])
      analytical_e22.append(epsil[1][1])
      analytical_e33.append(epsil[2][2])
   
    analytical_Sxx = []
    analytical_Syy = []
    analytical_Szz = []
    for sigma in analytical_sigmas:
      analytical_Sxx.append(sigma[0][0])
      analytical_Syy.append(sigma[1][1])
      analytical_Szz.append(sigma[2][2])

    ###PLOTTING
    formatter = ticker.FormatStrFormatter('$\mathbf{%g}$') 
    plt.figure(1)
    plt.clf()
    ax1 = plt.subplot(111)
    plt.subplots_adjust(right=0.75)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)    
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')    
    plt.title('AreniscaTest 11:\nUniaxial Strain J2 Plasticity (plot a)')
    plt.plot(np.array(analytical_e11),np.array(analytical_Sxx)/1e6,'--g',linewidth=lineWidth+1,label=str_to_mathbf('Analytical'))    
    plt.plot(np.array(exx)/1e-7,np.array(Sxx)/1e6,'-r',label=str_to_mathbf('Uintah'))
    plt.xlabel(str_to_mathbf('\epsilon_{A}'))
    plt.ylabel(str_to_mathbf('\sigma_{A} (Mpa)'))     
    plt.legend()    
    savePNG(save_path+'/Test11_verificationPlot_a','1280x960')      
    
    plt.figure(2)
    plt.clf()
    ax2 = plt.subplot(111)
    plt.subplots_adjust(right=0.75)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)    
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')     
    plt.title('AreniscaTest 11:\nUniaxial Strain J2 Plasticity (plot b)')    
    plt.plot(np.array(analytical_e11),np.array(analytical_Syy)/1e6,'--g',linewidth=lineWidth+1,label=str_to_mathbf('Analytical'))
    plt.plot(np.array(exx)/1e-7,np.array(Syy)/1e6,'-r',label=str_to_mathbf('Uintah'))
    plt.xlabel(str_to_mathbf('\epsilon_{A}'))
    plt.ylabel(str_to_mathbf('\sigma_{L} (Mpa)')) 
    plt.legend()
    savePNG(save_path+'/Test11_verificationPlot_b','1280x960')      

    plt.figure(3)
    plt.clf()
    ax3 = plt.subplot(111)
    plt.subplots_adjust(right=0.75)
    ax3.xaxis.set_major_formatter(formatter)
    ax3.yaxis.set_major_formatter(formatter)    
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')   
    plt.title('AreniscaTest 11:\nUniaxial Strain J2 Plasticity (plot c)')     
    plt.plot(analytical_times,np.array(analytical_e11),'-g',linewidth=lineWidth+1,label=str_to_mathbf('Analytical \epsilon_{xx}'))
    plt.plot(analytical_times,np.array(analytical_e22),'-r',linewidth=lineWidth+1,label=str_to_mathbf('Analytical \epsilon_{yy}'))
    plt.plot(analytical_times,np.array(analytical_e33),'-b',linewidth=lineWidth+1,label=str_to_mathbf('Analytical \epsilon_{zz}'))
    plt.legend()
    plt.xlabel(str_to_mathbf('Time (s)'))
    plt.ylabel(str_to_mathbf('\epsilon'))
    savePNG(save_path+'/Test11_verificationPlot_c','1280x960') 
    
    plt.figure(4)
    plt.clf()
    ax4 = plt.subplot(111)
    plt.subplots_adjust(right=0.75)
    ax4.xaxis.set_major_formatter(formatter)
    ax4.yaxis.set_major_formatter(formatter)    
    param_text = material_dict['material string']
    plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')   
    plt.title('AreniscaTest 11:\nUniaxial Strain J2 Plasticity (plot d)')     
    plt.plot(analytical_times,np.array(analytical_Sxx)/1e6,'-g',linewidth=lineWidth+1,label=str_to_mathbf('Analytical \sigma_{xx}'))
    plt.plot(analytical_times,np.array(analytical_Syy)/1e6,'-r',linewidth=lineWidth+1,label=str_to_mathbf('Analytical \sigma_{yy}'))
    plt.plot(analytical_times,np.array(analytical_Szz)/1e6,'-b',linewidth=lineWidth+1,label=str_to_mathbf('Analytical \sigma_{zz}'))
    plt.legend()
    plt.xlabel(str_to_mathbf('Time (s)'))
    plt.ylabel(str_to_mathbf('\sigma (Mpa)'))    
    savePNG(save_path+'/Test11_verificationPlot_d','1280x960') 
    
    if SHOW_ON_MAKE:
      plt.show()
  
  else:
    print '\nERROR: need working directory to post process this problem'  
  