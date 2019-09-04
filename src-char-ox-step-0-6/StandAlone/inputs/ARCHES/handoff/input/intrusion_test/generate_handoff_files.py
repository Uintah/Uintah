import numpy as np

def gaussian( stdx, stdy, A, x, y ): 

    return A * np.exp( -1. * ( x*x/(2*stdx*stdx) + y*y/(2*stdy*stdy) ))

A = 1.
stdx = .1
stdy = .1
nx = 5
ny = 5
shiftx = -2
shifty = -2
x = np.linspace(-.4, .4, nx)
y = np.linspace(-.4, .4, ny)

xv, yv = np.meshgrid(x,y, sparse=False, indexing='ij')

f = open('handoff_scalar.dat','w')
f.write('gaussian variable\n')
f.write('0.2 0.2\n')
f.write(np.str(nx*ny)+'\n')

for i in range(nx):
    for j in range(ny): 

        g = gaussian(stdx, stdy, A, xv[i][j], yv[i][j])
        to_write = '0 '+np.str(i+shiftx)+' '+np.str(j+shifty)+' '+np.str(g)+'\n'
        f.write(to_write)

f.close()

f = open('handoff_velocity.dat','w')
f.write('gaussian_variable\n')
f.write('0.2 0.2\n')
f.write(np.str(nx*ny)+'\n')

for i in range(nx):
    for j in range(ny): 

        g = gaussian(stdx, stdy, A, xv[i][j], yv[i][j])
        to_write = '0 '+np.str(i+shiftx)+' '+np.str(j+shifty)+' '+np.str(g)+' 1. 1.\n'
        f.write(to_write)

f.close()

