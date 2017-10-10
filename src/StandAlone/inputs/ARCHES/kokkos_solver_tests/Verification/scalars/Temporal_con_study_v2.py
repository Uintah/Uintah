import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def read_file(name_file):
    f = np.loadtxt(name_file)
    xf = []
    L = 1.0
    Nt = len(f[:,0])
    #dt = L/Nt
    t = np.linspace(0,L,Nt)
    dt = t[1]-t[0]
    return t, f[:,-1], dt

def mms(t):
    freq = 5.0
    int_r1 = int_src1(t,freq)
    int_r2 = int_src2(t)
    return int_totsrc(int_r1, int_r2,)

def int_src1(t,freq): 
    return -1.*np.cos(2*freq*np.pi*t)/(2.*freq*np.pi) + 1./(2.*freq*np.pi)
    
def int_src2(t): 
    return 1./3.*t**3
    
def int_totsrc(ir1,ir2): 
    return ir1 + ir2


def e(fe,f,dx,p =1.0):
    return (np.sum(abs(fe-f)**p)*dx)**(1./p)

def main(data = 'data_sine/', var_name ='cc_phi_central', var_mms = 'phi_mms', Nl =4, axis = 0, p =1, dt =1e-5):
    datname = [] 
    x =[]
    dx =[]
    f = []
    fe = []
    fmms = []
    L1 = []
    for i in range(Nl):
        ## mms
        data_mms = data + var_mms + '-t'+str(i)+'.txt'
        xe, fm, dxe = read_file(data_mms)
        ##fm =  mms(xe)
        ## variable
        datname.append(data + var_name + '-t'+str(i)+'.txt')
        x0, f0, dx0 = read_file(datname[i])

        #plt.figure()
        #plt.plot(x0,f0,'o')
        #plt.plot(xe,fm,'*r')
        #plt.show()    

        x.append(x0)
        f.append(f0)
        fe.append(fm)
        dx.append(dx0)


        e0 = e(f0,fm,dx0,p = p)
        L1.append(e0)
    L1 = np.array(L1) 
    x = np.array(dx)
    m, b, r_value, p_value, std_err = stats.linregress(np.log(dx),np.log(L1))
    print 'm = ',m,'b = ', b, 'r_value = ' , r_value  

#    plt.loglog(x,L1,'*--',label=var_name)
#    print x[0]
#    plt.plot(x1,f1,'*')
#    plt.plot(x2,f2,'s')
#    plt.figure()
#    plt.plot(x0,abs(f0-fm0),'o')

    

    return
if __name__ == "__main__":
    data = 'data/'
    Nl = 3 
    var_mms = 'phi_mms'

    var_name ='cc_phi_upwind'
    print var_name
    main(data,var_name,var_mms,Nl,p=1)



#    plt.show()
