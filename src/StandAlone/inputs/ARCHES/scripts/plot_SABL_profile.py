import matplotlib.pyplot as plt
import numpy as np

# A simple script to plot the profile of the StABL option
# given a set of parameters.

zo=.01   #roughness
zh=10.   #height of the freestream
velocity=1.1111  #freestream velocity
k=0.41           #von karman constant
ylow = 0.1       # > 0, lower bound
yhigh = 20.      # Top of domain
Npts = 100       # total number of points


kappa = k / np.log(zh/zo)
kappa *= kappa
ustar = np.sqrt( kappa * velocity * velocity )
y = np.linspace(ylow,yhigh,Npts)
vel = ustar / k * np.log( y / zo )

plt.plot(y, vel)
plt.ylabel('velocity')
plt.xlabel('height')
plt.show()
