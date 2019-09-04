import numpy as np
import sys

args=sys.argv

if args[1] == '--help' or args[1] =='-help':
    print ' \n'
    print ' Usage: python convert_pts.py <pts_file> <output_file> <conversion_factor> '
    print '        pts_file    : filename of the *.pts file'
    print '        output_file : filename to write to disk (add .pts on the end)'
    print '        conversion_factor : output_file = pts_file * conversion factor '
    print ' '
    sys.exit()

f = open(args[1],'r')

x=[]
y=[]
z=[]

for l in f:

    v=l.split()
    x.append(np.float(v[0]))
    y.append(np.float(v[1]))
    z.append(np.float(v[2]))

    #maxx = np.max(np.abs(np.float(v[0])), maxx)
    #maxy = np.max(np.abs(np.float(v[1])), maxy)
    #maxz = np.max(np.abs(np.float(v[2])), maxz)

convert = 1/args[3]

print 'Ranges: '
print 'x = ', np.min(x)/convert, ' - ', np.max(x)/convert
print 'y = ', np.min(y)/convert, ' - ', np.max(y)/convert
print 'z = ', np.min(z)/convert, ' - ', np.max(z)/convert

f.close()

if len(args) == 3:

  f = open(args[2], 'w')
  
  for i, item in enumerate(x): 
      to_write = np.str(x[i]/convert) + ' ' + np.str(y[i]/convert) + ' ' + np.str(z[i]/convert) + '\n'
      f.write(to_write)
  
  f.close()

