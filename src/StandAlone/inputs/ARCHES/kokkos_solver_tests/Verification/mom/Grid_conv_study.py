import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def read_file(name_file):
    f = np.loadtxt(name_file)
    Nx = int((np.shape(f)[0])**.5)
    f = np.reshape(f[:,3],(Nx,Nx),'F') # take the last column of phi and reshape
    xf = []
    L = 1.
    dx = L/Nx
    return  f, dx

def e(fe,f,dx,p =1.0):
    return (np.sum(abs(fe-f)**p)*dx)**(1./p)

def compute(data = 'data_sine/', var_name ='cc_phi_central', var_mms = 'phi_mms', Nl =4, axis = 0, p =1):
    datname = [] 
    x =[]
    dx =[]
    f = []
    fmms = []
    L1 = []
    for i in range(Nl):
        ## mms
        data_mms = data + var_mms + '-t'+str(i)+'.txt'
        fe, dxe = read_file(data_mms) 
        ## variable
        datname.append(data + var_name + '-t'+str(i)+'.txt')
        f0, dx0 = read_file(datname[i]) 
        f.append(f0)
        dx.append(dx0)
        # Normalization  in 2D... 
        DX = dx0*dx0
        e0 = e(f0,fe,DX,p = p)
        print e0
        L1.append(e0)

    L1 = np.array(L1) 
    dx = np.array(dx)
    print L1
    m, b, r_value, p_value, std_err = stats.linregress(np.log(dx),np.log(L1))
    print 'm = ',m,'b = ', b, 'r_value = ' ,r_value  

    plt.loglog(dx,L1,'*--',label=var_name)
#    print x[0]
#    plt.figure()
#    plt.plot(x[0],mms(x[0], wmms),'*')
#    plt.plot(x0,fm0,'o')
#    plt.plot(x1,f1,'*')
#    plt.plot(x2,f2,'s')
#    plt.figure()
#    plt.plot(x0,abs(f0-fm0),'o')

    

#    plt.show()    
    return
if __name__ == "__main__":
    plt.figure()
    data = 'data/'
    Nl = 3 
    var_mms = 'x_mms'

    var_name ='uVel'
    print var_name
    compute(data,var_name,var_mms,Nl)

    plt.show()
