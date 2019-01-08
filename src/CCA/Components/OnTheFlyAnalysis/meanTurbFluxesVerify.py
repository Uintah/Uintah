#!/usr/bin/env python

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
#______________________________________________________________________
#
#   This script computes the random samples from a multivariate normal distribution
#   and either plots the results or writes them to a file.  Four columns of 
#   numbers are written corresponding co 3 components of velocity (u,v,w) and a scalar 
#
#   The output is used for verification of the OnTheFly:meanTurbFlux analysis module
#
#______________________________________________________________________

from argparse import ArgumentParser

#parser = ArgumentParser(description='Process some integers.')
#parser.add_argument( "-f", "--file", dest="filename",
#                    help="write report to FILE", metavar="FILE")
#parser.add_argument( "-q", "--quiet",
#                    action="store_false", dest="verbose", default=True,
#                    help="don't print status messages to stdout" )

#args = parser.parse_args()
#______________________________________________________________________


nCells        = np.array( [4, 8, 6])
nPatches      = 1
nCellsPerPatch = np.divide( nCells, nPatches )
numSamples     = np.prod( nCellsPerPatch )

print ('nCellPerPatch %i' % numSamples)



doPlot     = False              # switch for plotting
doOutput   = True               # switch for file output
mean       = (10, 20 ,30, 0)    # mean values for u,v,w,s
stdDev     = 1                  # stdDev for all variables
filename   = 'test.out'

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

# Covariance of the data array
sigma = np.array( np.cov( data ) )

# Compute random samples from a multivariate normal distribution 
R = np.array( np.random.multivariate_normal(mu, sigma, numSamples) )

print('R=\n')
print(R)


# compute covariance of R it should be identical to 
# sigma
sigmaVerify = np.matrix( np.cov( data ) )


# Verification
diff = np.array( np.all( sigma - sigmaVerify ) );
if np.all( diff ==0 ):
  print ('Everything checks out\n covariance')
  print (sigmaVerify)
else:
  print ('There is a problem')


print ('\n__________________________________\n'
        ' Copy the file ( %s ) to the directory sus is in' % filename)





#__________________________________
#  Output file
if( doOutput ):
  mesg =  'This file contains 4 columns of random numbers from a multivariate normal distribution and\n'
  mesg += 'is used by OnTheFlyAnalysis:meanTurbFlux module for verification purposes.  Each cell in the\n'
  mesg += 'domain reads in one row from the file and populates a velocity (u,v,w) and a scalar label.\n'
  mesg += 'These labels are then processed by meanTurbFlux and the covariance for each plane should equal\n'
  mesg += 'the covariance computed by the script\n'
  mesg += '         u,                 v,                  w,                     scalar'

  np.savetxt( filename, R, fmt='%16.15e', delimiter=',', newline='\n',header=mesg)
  
#__________________________________
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


