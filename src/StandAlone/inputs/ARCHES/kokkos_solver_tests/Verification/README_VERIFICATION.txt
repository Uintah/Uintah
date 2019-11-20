The verification tests are run using python's unittest framework. 

Note that the setting of the SUSPATH environment variable
is required which will vary user to user. 
To do this, in your shell type (for bash):

export SUSPATH='....'


To your specific suspath. There has to be a better way of doing this part...

(note that the '-v' option (verbose) is not required )
To run ALL tests: 

python -v VerificationUnitTests.py

To run a SPECIFIC test: 

python -m unittest -v VerificationUnitTests.ArchesKokkosVerification.test_<NAME>

where <NAME> is one of the following: 

almgrenConv
almgrenDiff
almgrenMMSBC
xScalar
xScalarDiff
kokkosScalarRK1
kokkosScalarRK2
kokkosScalarRK3
xy2DScalar
xy2DScalarHandoff
xy2DScalarMMSBC

