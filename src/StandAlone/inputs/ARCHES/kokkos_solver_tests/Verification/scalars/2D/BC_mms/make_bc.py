## It uses to produce handoff files ..
##
import numpy as np

def write_scalar_file(var,name_file,name_scalar,resolution,X,Y,Z):
    [nX, nY] = np.shape(var)

    file_obj = open(name_file+ '.dat','w')
    file_obj.write(name_scalar+'\n')
    file_obj.write( repr(resolution)+' '+repr(resolution)+'\n' )
    file_obj.write( str(int(nX*nY))+'\n' )     # Write number of lines in file
    
    for i in range(nX):
        for j in range(nY):
            file_obj.write( str(X[i,j] )+' '+str(Y[i,j])+' '
            +str(Z[i,j])+' '+repr(var[i,j])+'\n' )        
    file_obj.close()
    return

def mms_function(x,y):
    A = 1.0
    f = 1.0 - A*np.cos(2.0*np.pi*x)*np.sin(2.0*np.pi*y) 
    return f


def mms_function1D(x):
    A = 1.0
    F = 1.0
    off_set = 1.0
    f = A*np.sin(F*2.0*np.pi*x) + off_set
    return f

def compute_BC(nz,ny):
    ##nz = 24 
    ##ny = 24
    L = 1.0
    dz = L/nz
    dy = L/ny
    z =  range(nz)
    xp = np.loadtxt('mama.out')
    y =range(ny)
    Z, Y = np.meshgrid(z, y)
    Zp = np.array(Z)*dz + dz/2.
    Yp = np.array(Y)*dy + dy/2.
    X = -np.ones(np.shape(Zp))
    xl =-dz/2. 
    xr =1.0+ dz/2.
    v = mms_function(xl,Yp)
    v1 = mms_function(xr,Yp)
    #write_scalar_file(v,'x_left','phi',dy,X,Y,Z) 
    #write_scalar_file(v,'x_rigth','phi',dy,-nz*X,Y,Z) 

    vt = np.vstack((v,v1))
    Xt = np.vstack((X,-X))
    Yt = np.vstack((Y,Y))
    Zt = np.vstack((Z,Z))
      
    write_scalar_file(vt,'x_lr'+str(ny),'phi',dy,Xt,Yt,Zt) 
    write_scalar_file(vt,'x_lr','phi',dy,Xt,Yt,Zt) 

    return

if __name__ == "__main__":
   nx =96 
   compute_BC(nx,nx)

   nx =48 
   compute_BC(nx,nx)

   nx =24 
   compute_BC(nx,nx)

