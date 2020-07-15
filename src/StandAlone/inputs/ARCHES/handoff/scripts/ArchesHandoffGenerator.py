def get_dims(filename): 
    
    import numpy as np 
    
    """ Returns the changing dims in a handoff file."""
    
    myfile = np.loadtxt(filename,skiprows=3)
    I = myfile[:,0]
    J = myfile[:,1]
    K = myfile[:,2]

    mini = np.min(I)
    maxi = np.max(I)
    minj = np.min(J)
    maxj = np.max(J)
    mink = np.min(K)
    maxk = np.max(K)

    if maxi - mini == 0: 
        return 1,2
    elif maxj - minj == 0: 
        return 0,2
    else: 
        return 0,1

def getInterpolator(filename, refine_ratio):
    
    import numpy as np
    from scipy import interpolate 
    import matplotlib.pyplot as plt

    """ Returns an interpolated field at a specified refinement ratio 
        Also returns original x,y,data values and the new Xnew, Ynew 
        locations of the coarsened data. """
    
    dim1, dim2 = get_dims(filename)
    var = np.loadtxt(filename, skiprows=3)
    f = open(filename,'r')
    for i, l in enumerate(f):
        if i == 1: 
            dx = np.float(l.split()[0])
            dy = np.float(l.split()[1])
            break
    f.close()

    #which dim is changing fastest:
    fastdim = -1
    diffx = var[1,dim1]-var[0,dim1]
    diffy = var[1,dim2]-var[0,dim2]
        
    if diffx > 0: 
        fastdim = dim1
    else:
        fastdim = dim2

    #how many cycles in the fastdim until we repeat:
    fastdim_cycle = -1
    for i, v in enumerate(var): 
        if i > 0: 
            if v[fastdim] == var[0,fastdim]: 
                fastdim_cycle = i
                break

    if fastdim == -1 or fastdim_cycle == -1: 
        print('Error: cannot determine the fastest changing dimension.')
        sys.exit()

    startx = var[0,dim1]
    starty = var[0,dim2]
    endx = var[len(var)-1,dim1]
    endy = var[len(var)-1,dim2]

    xi = np.arange(startx,endx+1,1)
    yi = np.arange(starty,endy+1,1)

    x = xi*dx+dx/2
    y = yi*dy+dy/2

    Nx = len(x)
    Ny = len(y)

    newNx = round(Nx/refine_ratio)
    newNy = round(Ny/refine_ratio)

    newX = np.linspace(x[0],x[Nx-1],newNx)
    newY = np.linspace(y[0],y[Ny-1],newNy)

    xx, yy = np.meshgrid(x,y)

    var_2d = var[:,3].reshape((Nx,Ny))

    return interpolate.interp2d(xx, yy, var_2d, kind='linear'), x, y, newX, newY, var_2d
