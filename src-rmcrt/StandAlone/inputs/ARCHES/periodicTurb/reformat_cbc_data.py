import numpy as np

# This utility takes the traditional form of the cbc data: 
# 
#  ni nj nk
#  u[0] v[0] w[0]
#  u[1] v[1] w[1]
#  .....
# 
# and formats it to 
#
#  ni nj nk
#  i j k u[0] v[0] w[0]
#  i j k u[0] v[0] w[0]
#  ....
#
# which is intended to be a more generic form. 

f = open('cbc', 'r')
fout = open('output.out', 'w')

nx = 0
ny = 0
nz = 0


for i, line in enumerate(f): 

    L = line.split(' ')

    if i == 0: 
        nx = np.int(L[0])
        ny = np.int(L[1])
        nz = np.int(L[2])
        N = nx*ny*nz
        u = np.zeros(N)
        v = np.zeros(N)
        w = np.zeros(N)
    else: 
        u[i-1] = np.float(L[0])
        v[i-1] = np.float(L[1])
        w[i-1] = np.float(L[2])

II = 0
for k in range(0,nz):
    for j in range(0,ny): 
        for i in range(0,nx): 

            line = np.str(i)+' '+np.str(j)+' '+np.str(k)+' '+np.str(u[II])+' '+np.str(v[II])+' '+np.str(w[II])+'\n'
            II += 1
            fout.write(line)


f.close()
fout.close()

