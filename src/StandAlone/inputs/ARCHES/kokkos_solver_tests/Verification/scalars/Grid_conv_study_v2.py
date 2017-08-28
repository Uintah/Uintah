import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def read_file(name_file, dire = 0):
    f = np.loadtxt(name_file)
    xf = []
    L = 1.
    dx = L/len(f[:,0])
    x = f[:,dire]*dx 
    return x, f[:,-1], dx

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
        xe, fe, dxe = read_file(data_mms,axis) 
        ## variable
        datname.append(data + var_name + '-t'+str(i)+'.txt')
        x0, f0, dx0 = read_file(datname[i],axis) 
        x.append(x0)
        f.append(f0)
        dx.append(dx0)
        e0 = e(f0,fe,dx0,p = p)
        L1.append(e0)

    L1 = np.array(L1) 
    dx = np.array(dx)

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
   compute()
