import math
import os
import sys
from scipy import *

Nmats=5
Rmin = 0.001 
Rmax = 0.00225
# Decreased gap
gap = 0.0005
# Shortened length
Ymin = -0.2
Ymax = 0.0
Ljet = Ymax - Ymin
# Velocity
Vmax=8300.0
Vmin =1350.0
Rslope=(Rmax-Rmin)/Ljet
Vslope=(Vmax-Vmin)/Ljet
# Standoff 
standoff = -Rmax - (gap/2.0)
# Calculate Time
Time=0.0
# StandoffTime = Time - (standoff/Vmax)
# Elevated Temperature
Temp=700.0
# Reduced Density
###density = array([0.0,19300.0,8930.0,19300.0,11350.0,19300.0])
density = array([0.0,19300.0/2.0,8930.0/2.0,19300.0/2.0,11350.0/2.0,19300.0/2.0])
volume = zeros(Nmats+1,dtype='f') 
mass = zeros(Nmats+1,dtype='f') 
TotVol=0.0
TotMass=0.0

outputfile=open('jet.txt','w')
outputfile.write('Jet Length = '+str(Ljet)+' m \n')
outputfile.write('Jet Radius = '+str(Rmin)+' to '+str(Rmax)+' m \n')
outputfile.write('Jet Velocity Range = '+str(Vmax)+' to '+str(Vmin)+' m/s \n')
outputfile.close()

i=1
Nspheres=int(1)
while i <= Nmats:
	outputfile='outputfile'+str(i)
	outputfile=open('spheres'+str(i)+'.xml','w')
	outputfile.write("<?xml version='1.0' encoding='ISO-8859-1'?>"+"\n")
	outputfile.write("<Uintah_Include> \n")
	outputfile.close()
	i+=1

outputfile='outputfile'+str(i)
outputfile=open('InsertParticle.dat','w')
outputfile.close()

y=Ymax
Rsphere = -Rslope*y + Rmin 
ytmp = y - Rsphere + (gap/2.0)
vtmp = Vslope*ytmp + Vmax 

outputfile=open('jet.txt','a')
outputfile.write('Tip Velocity = '+str(vtmp)+' m/s \n')
outputfile.close()

while (y > Ymin + Rsphere):
	i=1
	while i<= Nmats and (y > Ymin + Rsphere):
		Rsphere = -Rslope*y + Rmin 
		vol=(4.0/3.0)*math.pi*(Rsphere**3)
		volume[i] += vol
		TotVol += vol 
		standoff = -Rsphere - (gap/2.0)
	        y = y + standoff 
	       	yvelocity = Vslope*y + Vmax 
		Time = -y/yvelocity
		outputfile='outputfile'+str(i)
		outputfile=open('spheres'+str(i)+'.xml','a')
	       	outputfile.write('  <geom_object> \n')
	        outputfile.write('    <sphere label = "sphere_'+str(Nspheres)+'"> \n')
	        outputfile.write('      <origin>[0.0,'+str(standoff)+',0.0]</origin> \n')
	        outputfile.write('      <radius>'+str(Rsphere)+'</radius> \n')
	        outputfile.write('    </sphere> \n')
	        outputfile.write('    <res>[2,2,1]</res> \n')
	        outputfile.write('    <velocity>[0.0,'+str(yvelocity)+',0.0]</velocity> \n')
	        outputfile.write('    <temperature>'+str(Temp)+'</temperature> \n')
	        outputfile.write('    <!-- <InsertTime>'+str(Time)+'</InsertTime> -->\n')
	        outputfile.write('    <color>'+str(Nspheres)+'</color>\n')
	        outputfile.write('  </geom_object> \n')
	        y = y + standoff
		outputfile.close()
                outputfile=open('InsertParticle.dat','a')
                outputfile.write(str(Time)+'  '+str(Nspheres)+'  '+str(standoff)+'  '+str(yvelocity)+'\n')
		outputfile.close()
		Nspheres +=1
		i +=1

outputfile=open('jet.txt','a')
outputfile.write('Tail Velocity = '+str(yvelocity)+' m/s \n')
outputfile.write('Number of Materials = '+str(Nmats)+'\n')
outputfile.close()

i=1
while i <= Nmats:
	outputfile='outputfile'+str(i)
	outputfile=open('spheres'+str(i)+'.xml','a')
	outputfile.write('</Uintah_Include> \n')
	outputfile.close()

	outputfile=open('jet.txt','a')
	outputfile.write('Density '+str(i)+' = '+str(density[i])+' kg/m^3 \n')
	outputfile.write('Volume '+str(i)+' = '+str(volume[i])+' m^3 \n')
	mass[i] = volume[i]*density[i]
	outputfile.write('Mass '+str(i)+' = '+str(mass[i])+' kg \n')
	outputfile.close()

	TotMass += mass[i]
	i+=1

outputfile=open('jet.txt','a')
outputfile.write('Number of Spheres = '+str(Nspheres)+'\n')
outputfile.write('Gap between Spheres = '+str(gap)+' m \n')
outputfile.write('Total Volume = '+str(TotVol)+' m^3 \n')
outputfile.write('Total Mass = '+str(TotMass)+' kg \n')
outputfile.write('Average Density = '+str(TotMass/TotVol)+' kg/m^3 \n')
outputfile.write('% 1 wt = '+str(100.*(mass[1]+mass[3]+mass[5])/TotMass)+'\n')
outputfile.write('% 2 wt = '+str(100.*(mass[2])/TotMass)+'\n')
outputfile.write('% 3 wt = '+str(100.*(mass[4])/TotMass)+'\n')
outputfile.close()
