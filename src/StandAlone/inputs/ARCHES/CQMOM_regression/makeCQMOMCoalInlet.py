import numpy as np

# limit to two quad nodes in d_p, two in u
# 30 total moments

#D_p  [  1.50000000e-05   7.50000000e-05]
#M_p  [  2.29728963e-12   2.87161203e-10]
#RC  [  2.10523622e-12   2.63154527e-10]
#ash  [  1.92053413e-13   2.40066766e-11]
#char  [ 0.  0.]
#E  [ -1.95894490e-05  -2.44868113e-03]
#w [224815360621.0 1238295053.01]

#initialization with points
#PSD diameters
d1 = 15
d2 = 75

#PSD weights
w1 = 224815360621.0
w2 = 1238295053.01

#intial raw coal
# coal1 corresponds to d1 mass
coal1 = 2.10523622e-12
coal2 = 2.63154527e-10

#initial char
char1 = 0
char2 = 0

#inital entalhpy
e1 = -1.95894490e-05
e2 = -2.44868113e-03

#intial u-vel 
#needs variance to maintain other values
u0 = 10
ep = .1

#initial v-vel
#can add variance if desired- variance conditioned on U
v0 = 0
epV = 0

#inital w-vel
w0 = 0
epW = 0

d = np.array([d1,d1,d2,d2])* 1.0e-6
weight = np.array([w1/2,w1/2,w2/2,w2/2])
coal = np.array([coal1,coal1,coal2,coal2])
char = np.array([char1,char1,char2,char2])
enthalpy = np.array([e1,e1,e2,e2])
u = np.array([u0-ep,u0+ep,u0-ep,u0+ep])
v = np.array([v0-epV,v0-epV,v0+epV,v0+epV])
w = np.array([w0-epW,w0+epW,w0-epW,w0+epW])

i = np.array([0,1,2,3,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]);
j = np.array([0,0,0,0,1,1,2,2,3,3,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]);
k = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
l = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]);
m = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0]);
o = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0]);
p = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]);
n = np.size(i)

# moments for 3 x 2 initialization for better PSD description - needs 3 nodes set for dp,coal,enthalpy,etc
#i = np.array([0,1,2,3,4,5,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]);
#j = np.array([0,0,0,0,0,0,1,1,1,2,2,2,3,3,3,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1]);
#k = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
#l = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]);
#m = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]);
#o = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0]);
#p = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]);
#n = np.size(i)

moments = np.zeros(n)

for ii in range(0,n):
  moments[ii] = np.sum(weight * d**i[ii] * u**j[ii] * v**k[ii] * w**l[ii] * coal**m[ii] * char**o[ii] * enthalpy**p[ii] )



print 'Insert into CQMOM Tags'
print '--------------------------------------'
print '        <NumberInternalCoordinates> 7 </NumberInternalCoordinates>'
print '        <QuadratureNodes> [2,2,1,1,1,1,1] </QuadratureNodes>'
for ii in range (0,n):
  print '        <Moment> <m> [%d,%d,%d,%d,%d,%d,%d] </m> </Moment>' % (i[ii],j[ii],k[ii],l[ii],m[ii],o[ii],p[ii] )

print 'Insert into Outlet BC'
print '--------------------------------------'
for ii in range (0,n):
  print '        <BCType label="m_%d%d%d%d%d%d%d" var="Neumann">' % (i[ii],j[ii],k[ii],l[ii],m[ii],o[ii],p[ii] )
  print '          <value> 0.0 </value>'
  print '        </BCType>'

print 'Insert into Wall BC'
print '--------------------------------------'
for ii in range (0,n):
  print '        <BCType label="m_%d%d%d%d%d%d%d" var="ForcedDirichlet">' % (i[ii],j[ii],k[ii],l[ii],m[ii],o[ii],p[ii] )
  print '          <value> 0.0 </value>'
  print '        </BCType>'

print 'Insert into Inlet BC'
print '--------------------------------------'
for ii in range (0,n):
  print '        <BCType label="m_%d%d%d%d%d%d%d" var="ForcedDirichlet">' % (i[ii],j[ii],k[ii],l[ii],m[ii],o[ii],p[ii] )
  print '          <value> %.14e </value>' % (moments[ii])
  print '        </BCType>'




# try initialization with continuous normal or log normal functions?
# required transported moments are the same
# calculate required moments with integrated values m_ijk... = \int \int \int f(u,v,w) u^i v^j w^k ... du dv dw ...


