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

def getInterpolator(filename, refine_ratio, spline_order=(3,3)):
    
    import numpy as np
    from scipy import interpolate 
    import matplotlib.pyplot as plt

    """ Returns an interpolated field at a specified refinement ratio 
        Also returns original x,y,data values and the new Xnew, Ynew 
        locations of the coarsened data. """
    
    #dim1, dim2 = get_dims(filename)
    dim1 = 0
    dim2 = 1
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

    var_2d = var[:,2].reshape((Nx,Ny))

    import matplotlib.pyplot as plt

    plt.contourf(xx,yy,var_2d)

    #interpolator = interpolate.interp2d(xx, yy, var_2d, kind='linear')
    interpolator = interpolate.SmoothBivariateSpline(xx.reshape(Nx**2),yy.reshape(Ny**2),var[:,2],kx=spline_order[0],ky=spline_order[1])

    new_var = np.zeros((newNx,newNy))

    for i in range(newNx): 
        for j in range(newNy): 

            new_var[i,j] = interpolator(newX[i],newY[j])

    print(' ----- filename: ',filename,' -----')
    print('Min of orig variable was :{}'.format(np.min(var_2d)))
    print('Min of interp variable is :{}'.format(np.min(new_var)))
    print('Max of orig variable was :{}'.format(np.max(var_2d)))
    print('Max of interp variable is :{}'.format(np.max(new_var)))

    newXX, newYY = np.meshgrid(newX,newY)
    print(np.shape(newX))
    plt.contour(newXX,newYY,new_var,linewidths=3.5,linestyle='dotted')
    plt.title(filename)
    plt.show()

    return interpolator, x, y, newX, newY, var_2d
