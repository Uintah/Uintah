import math
import os
import sys

Res=0.0005
Width = 0.0762 
HalfWidth = Width/2.0
FourthWidth = Width/4.0
Thickness = 0.0127
gap = 0.000
zmin = 0.0
zmax = 0.381
Ltarget = zmax - zmin
print 'Target Length= ',Ltarget,'\n'
Nplates = int(Ltarget/(Thickness+gap))
print 'Number of Plates= ',Nplates,'\n'


outputfile=open('plate1.xml','w')
outputfile.write("<?xml version='1.0' encoding='ISO-8859-1'?>"+"\n")
outputfile.write("<Uintah_Include>"+"\n")

outputfile2=open('plate2.xml','w');
outputfile2.write("<?xml version='1.0' encoding='ISO-8859-1'?>"+"\n")
outputfile2.write("<Uintah_Include>"+"\n")

z=zmin
for i in range(0,Nplates,2):
        outputfile.write('  <geom_object> \n')
        outputfile.write('   <difference> \n')
        outputfile.write('    <box label = "plateL_'+str(i+1)+'"> \n')
        outputfile.write('      <min>[-'+str(HalfWidth)+','+str(z)+',0.0]</min> \n')
        outputfile.write('      <max>['+str(HalfWidth)+','+str(z+Thickness)+','+str(Res)+']</max> \n')
        outputfile.write('    </box> \n')
        outputfile.write('    <box label = "plateH_'+str(i+1)+'"> \n')
        outputfile.write('      <min>[-'+str(FourthWidth)+','+str(z)+',0.0]</min> \n')
        outputfile.write('      <max>['+str(FourthWidth)+','+str(z+Thickness)+','+str(Res)+']</max> \n')
        outputfile.write('    </box> \n')
        outputfile.write('   </difference> \n')
        outputfile.write('   <res>[1,1,1]</res> \n')
        outputfile.write('   <velocity>[0.0,0.0,0.0]</velocity> \n')
        outputfile.write('   <temperature>294</temperature> \n')
        outputfile.write('   <color>-1</color> \n')
        outputfile.write('  </geom_object> \n')
        outputfile.write(' \n')

        outputfile.write('  <geom_object> \n')
        outputfile.write('    <box label = "plateH_'+str(i+1)+'"/> \n')
        outputfile.write('    <res>[2,2,1]</res> \n')
        outputfile.write('    <velocity>[0.0,0.0,0.0]</velocity> \n')
        outputfile.write('    <temperature>294</temperature> \n')
        outputfile.write('    <color>-1</color> \n')
        outputfile.write('  </geom_object> \n')
        outputfile.write(' \n')
        
        z=z+(Thickness+gap)

        outputfile2.write('  <geom_object> \n')
        outputfile2.write('    <difference> \n')
        outputfile2.write('    <box label = "plateL_'+str(i+2)+'"> \n')
        outputfile2.write('      <min>[-'+str(HalfWidth)+','+str(z)+',0.0]</min> \n')
        outputfile2.write('      <max>['+str(HalfWidth)+','+str(z+Thickness)+','+str(Res)+']</max> \n')
        outputfile2.write('    </box> \n')
        outputfile2.write('    <box label = "plateH_'+str(i+2)+'"> \n')
        outputfile2.write('      <min>[-'+str(FourthWidth)+','+str(z)+',0.0]</min> \n')
        outputfile2.write('      <max>['+str(FourthWidth)+','+str(z+Thickness)+','+str(Res)+']</max> \n')
        outputfile2.write('    </box> \n')
        outputfile2.write('    </difference> \n')
        outputfile2.write('   <res>[1,1,1]</res> \n')
        outputfile2.write('   <velocity>[0.0,0.0,0.0]</velocity> \n')
        outputfile2.write('   <temperature>294</temperature> \n')
        outputfile2.write('   <color>-1</color> \n')
        outputfile2.write('  </geom_object> \n')
        outputfile2.write(' \n')

        outputfile2.write('  <geom_object> \n')
        outputfile2.write('    <box label = "plateH_'+str(i+2)+'"/> \n')
        outputfile2.write('    <res>[2,2,1]</res> \n')
        outputfile2.write('    <velocity>[0.0,0.0,0.0]</velocity> \n')
        outputfile2.write('    <temperature>294</temperature> \n')
        outputfile2.write('    <color>-1</color> \n')
        outputfile2.write('  </geom_object> \n')
        outputfile2.write(' \n')

        z=z+(Thickness+gap)

outputfile.write('</Uintah_Include> \n')

outputfile2.write('</Uintah_Include> \n')

