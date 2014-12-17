#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#Plotting stuff below
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import ticker 

SHOW_ON_MAKE = False

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


###TEST PLOTTING WITH LATEX###
if __name__=="__main__":
  print "Testing python matplotlib using LATEX"
  endT = 4*np.pi
  times = np.linspace(0,endT,200)
  vals1 = np.sin(times)
  vals2 = np.cos(times)    
  
  ###PLOTTING
  formatter = ticker.FormatStrFormatter('$\mathbf{%g}$') 
  plt.figure(1)
  plt.clf()
  plt.subplots_adjust(right=0.75)
  param_text = "RANDOM TEXT\n   abc\n   xyz\n   ijk\n   123\n   !*="
  plt.figtext(0.77,0.70,param_text,ha='left',va='top',size='x-small')
  
  #vals1
  ax2 = plt.subplot(212)
  #without rotation
  plt.plot([0,endT],[0,0],'-b')
  #simulation results
  plt.plot(times,vals1,'-r')  
  #guide line
  plt.plot([0,endT],[-1,-1],'--g')
  #labels and limits
  ax2.set_ylim(-70,10)
  plt.grid(True)
  ax2.xaxis.set_major_formatter(formatter)
  ax2.yaxis.set_major_formatter(formatter)
  plt.ylabel(str_to_mathbf('\sigma_{11} (Units)'))
  plt.xlabel(str_to_mathbf('Time (s)'))
  
  #vals2
  ax1 = plt.subplot(211,sharex=ax2,sharey=ax2)
  plt.setp(ax1.get_xticklabels(), visible=False)
  #without rotation
  plt.plot([0,endT],[0,0],'-b',label='line1') 
  #simulation results
  plt.plot(times,vals2,'-r',label='vals')
  #guide lines
  plt.plot([0,endT],[1,1],'--g',label='line2') 
  
  #labels
  ax1.set_ylim(-1.1,1.1)
  plt.grid(True)
  ax1.xaxis.set_major_formatter(formatter)
  ax1.yaxis.set_major_formatter(formatter)
  ax1.set_yticks([-1,-0.5,0.0,0.5,1.0])  
  plt.ylabel(str_to_mathbf('\epsilon_{22} (Units)')) 
  plt.title('Matplotlib Test:\nChecking Latex Works')
  plt.legend()
  savePNG("testPlot",'1280x960')
  plt.show()


