#!/usr/bin/env python

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
#______________________________________________________________________
#
#   This script computes the random samples from a multivariate normal distribution
#   and either plots the results or writes them to a file.  Four columns of 
#   numbers are written corresponding co 3 components of velocity (u,v,w) and a scalar (s)
#
#   The output is used for verification of the OnTheFly:meanTurbFlux analysis module
#
#______________________________________________________________________

#______________________________________________________________________


nPlaneCells   = np.array( [80, 80, 1])     # x, y, z

nPatches      = 1
nCellsPerPatch = np.divide( nPlaneCells, nPatches )
numSamples     = int( np.prod( nCellsPerPatch ) )

print ('nCellPerPatch %i' % numSamples)


doPlot     = True               # switch for plotting
doOutput   = True               # switch for file output
mean       = (10, 20 ,30, 0)    # mean values for u, v, w, scalar
stdDev     = 1                  # stdDev for all variables
filename   = 'testDistribution.txt'

#______________________________________________________________________
#  Create an array of random numbers centered around a mean with a 
#  standard deviation of stdDev

data = np.array( [ np.random.normal( mean[0], stdDev, numSamples), 
                   np.random.normal( mean[1], stdDev, numSamples),
                   np.random.normal( mean[2], stdDev, numSamples),
                   np.random.normal( mean[3], stdDev, numSamples) ]  )
#print('data=\n')
#print (data)

# compute the mean for each column in the data array
mu    = np.array( np.mean( data, axis=1 ) )
print('mu\n')
print(mu)

# Population Covariance of the data array
# see: https://en.wikipedia.org/wiki/Covariance#Calculating_the_sample_covarianc
sigma = np.array( np.cov( data, rowvar=True, bias=True ) )


# Compute random samples from a multivariate normal distribution 
R = np.array( np.random.multivariate_normal(mu, sigma, numSamples) )
print('\nR=\n')
print(R)

R2 = np.tile( R, (2,1))
print('\nR2=\n')
print(R2)


# Population covariance of R 
sigmaR = np.array( np.cov( R, rowvar=False, bias=True ) )
print('\nsigmaR\n')
print(sigmaR)


#______________________________________________________________________
#  Output
if( doOutput ):

  print ('\n__________________________________\n'
          ' The file ( %s ) contains the multivariate normal distribution' % filename)

  hdr =  'This file contains 4 columns of random numbers from a multivariate normal distribution and\n'
  hdr += 'is used by the OnTheFlyAnalysis:meanTurbFlux module for verification purposes.  Each cell in the\n'
  hdr += 'domain reads in one row from the file and populates a velocity (u,v,w) and a scalar (s).\n'
  hdr += 'These labels are then processed by meanTurbFlux and the covariance for each plane should equal\n'
  hdr += 'the covariance computed by the script\n'
  hdr += '         u,                 v,                  w,                     scalar'
  
  # prepend line numbers to R
  length  = R.shape[0]
  indices = np.arange( length )
  newR    = np.column_stack( (indices, R) )

  np.savetxt( filename, newR, fmt='%i, %16.15e, %16.15e, %16.15e, %16.15e', newline='\n', header=hdr)

  # covariance
  print ('\n__________________________________\n'
        ' The file ( covariance.txt ) contains the covariance')
        
  hdr =  'This file contains the population covariance of a multivariate normal distribution and\n'
  hdr += 'is used by OnTheFlyAnalysis:meanTurbFlux module for verification purposes.\n'
  hdr += ' See https://en.wikipedia.org/wiki/Covariance#Calculating_the_sample_covariance\n'  
  desc = np.array([ ["u'u'^bar", "v'u'^bar", "w'u'^bar", "s'u'^bar" ],
                   [ "u'v'^bar", "v'v'^bar", "w'v'^bar", "s'v'^bar" ],
                   [ "u'w'^bar", "v'w'^bar", "w'w'^bar", "s'w'^bar" ],
                   [ "u's'^bar", "v's'^bar", "w's'^bar", "s's'^bar" ] ])
  print(desc)
  print(sigmaR)

  # glue desc and sigmaR together.
  arry = np.zeros( desc.shape, dtype=[('var1', 'S8'), ('var2', float)])
  arry['var1'] = desc
  arry['var2'] = sigmaR
  np.savetxt('covariance.txt', arry, fmt="%10s", header=hdr)


#______________________________________________________________________
#  plotting
if( doPlot ):
  xs = R[:,0]
  ys = R[:,1]
  zs = R[:,2]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  ax.scatter( xs, ys, zs, c='r', marker='o')
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  plt.show()
