#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

#Shorthand character combinations
tab = '\t'
endl = '\n'
One = '1.0'
Zed = '0.0'
dblZed = Zed+tab+Zed
rotation = Zed+tab+Zed+tab+Zed+tab+Zed+endl

#output numerical precision
precision = '1.8e'

#Write each leg to file
def linearInterpEpsil(F,tInit,delT,delEpsil_11,delEpsil_22,delEpsil_33,epsil11_init=0.0,epsil22_init=0.0,epsil33_init=0.0,numPnts = 10000):
  epsilSlope_11 = delEpsil_11/delT
  epsilSlope_22 = delEpsil_22/delT
  epsilSlope_33 = delEpsil_33/delT
  subDelT = delT/numPnts
  time = tInit
  epsil_11 = epsil11_init
  epsil_22 = epsil22_init
  epsil_33 = epsil33_init
  for i in range(numPnts):
    time += subDelT
    epsil_11 += subDelT*epsilSlope_11
    epsil_22 += subDelT*epsilSlope_22
    epsil_33 += subDelT*epsilSlope_33
    F11 = np.exp(epsil_11)
    F22 = np.exp(epsil_22)
    F33 = np.exp(epsil_33)
    F.write(format(time,precision)+tab+format(F11,precision)+tab+dblZed+tab+Zed+tab+format(F22,precision)+tab+Zed+tab+dblZed+tab+format(F33,precision)+tab+rotation)

#Open file
F = open("AreniscaTest_10_PrescribedDeformation.inp","w")
#Initial State
F.write(Zed+tab+One+tab+dblZed+tab+Zed+tab+One+tab+Zed+tab+dblZed+tab+One+tab+rotation)
#First Leg
t = 0.0
delT = 1.0
delEpsil_11 = -0.003
delEpsil_22 = -0.003
delEpsil_33 =  0.006
linearInterpEpsil(F,t,delT,delEpsil_11,delEpsil_22,delEpsil_33)
#Second Leg
t = 1.0
delT = 1.0
epsil11_init = -0.003
epsil22_init = -0.003
epsil33_init =  0.006
delEpsil_11 = -0.0073923
delEpsil_22 =  0.003
delEpsil_33 =  0.0043923
linearInterpEpsil(F,t,delT,delEpsil_11,delEpsil_22,delEpsil_33,epsil11_init,epsil22_init,epsil33_init)
#Close file handle
F.close