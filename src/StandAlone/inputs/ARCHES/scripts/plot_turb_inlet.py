import numpy as np
import matplotlib.pyplot as plt
import sys

def usage():
    print('Usage: python plot_turb_inlet.py FILENAME DENSITY CELL_FACE_AREA\n')
    print('       Generates a series of PDF files showing the')
    print('       u,v,w values of the turbulent inlet.  \n')
    exit()

if len(sys.argv) != 4:
    usage()

if sys.argv[1] == "--help" or sys.argv[1] == "-help":
    usage()

filename = sys.argv[1]

data = np.loadtxt(filename,skiprows=17); 

f = open(filename,'r')
Nx = 0
Ny = 0
Nt = 0

for i in range(0,17): 

    line = f.readline()
    
    if i == 1: 
        sline = line.split(' ')
        Nx=np.int(sline[1])
        Ny=np.int(sline[3])

        

    if i == 13: 
        Nt = np.int(line)

Ngrid = Nx*Ny
Nrepeat = Nt/Ngrid

istart = 0
iend = Ngrid

rho = np.float(sys.argv[2])
area = np.float(sys.argv[3])

usum = np.zeros([Nx,Ny])
vsum = np.zeros([Nx,Ny])
wsum = np.zeros([Nx,Ny])

for i in range(0,Nrepeat):

    print('Creating plot for time slice: '+np.str(i))
    plt.figure()
    plt.subplot(311)
    u = data[istart:iend,3].reshape([Nx,Ny])

    usum += u
    plt.contourf(u, cmap='plasma', interpolation='nearest')
    plt.title('u')
    plt.colorbar()
    plt.subplot(312)
    v = data[istart:iend,4].reshape([Nx,Ny])
    vsum += v
    plt.contourf(v, cmap='plasma', interpolation='nearest')
    plt.title('v')
    plt.colorbar()
    plt.subplot(313)
    w = data[istart:iend,5].reshape([Nx,Ny])

    wsum += w
    plt.contourf(w, cmap='plasma', interpolation='nearest')
    plt.title('w')
    plt.colorbar()
    plt.savefig('turbulent_inlet_prof_t'+np.str(i)+'.tiff')

    uave = np.mean(u)
    vave = np.mean(v)
    wave = np.mean(w)
    mag = np.sqrt(u*u + v*v + w*w)

    print('   Average u,v,w = '+np.str(uave)+', '+np.str(vave)+', '+np.str(wave))
    print('   Max u,v,w =     '+np.str(np.amax(u))+', '+np.str(np.amax(v))+', '+np.str(np.amax(w)))
    print('   Mdot = '+np.str( np.sum(v*area*rho)  ))
    print('   Mean mag = '+np.str(np.mean(mag)))

    plt.figure()
    plt.imshow(mag, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.savefig('magnitude_t'+np.str(i)+'.tiff')

    plt.close('all')

    istart += Ngrid
    iend += Ngrid

uhat = usum/(Nrepeat-1)
vhat = vsum/(Nrepeat-1)
what = wsum/(Nrepeat-1)

print('Time ave, spatial mean of u: '+np.str(np.mean(usum)))
print('Time ave, spatial mean of v: '+np.str(np.mean(vsum)))
print('Time ave, spatial mean of w: '+np.str(np.mean(wsum)))

istart = 0
iend = Ngrid

#fluctuations
up = np.zeros([Nx,Ny])
vp = np.zeros([Nx,Ny])
wp = np.zeros([Nx,Ny])

print('Now computing the U\' variables')
for i in range(0,Nrepeat):

    print('Working on slice: '+np.str(i))
    u = data[istart:iend,3].reshape([Nx,Ny])
    up += uhat - u 

    v = data[istart:iend,3].reshape([Nx,Ny])
    vp += vhat - v 

    w = data[istart:iend,3].reshape([Nx,Ny])
    wp += what - w 

    istart += Ngrid
    iend += Ngrid

up /= (Nrepeat-1)
vp /= (Nrepeat-1)
wp /= (Nrepeat-1)

Urms_prime = np.sqrt(1./3.*(up*up+vp*vp+wp*wp))
Urms = np.sqrt(1./3.*(uhat*uhat+vhat*vhat+what*what))

I = Urms_prime/Urms

plt.contourf(I)
plt.colorbar()
plt.savefig('TurbulenceIntensity.tiff')

print('Ave Turb Intensity: '+np.str(np.mean(I)))

