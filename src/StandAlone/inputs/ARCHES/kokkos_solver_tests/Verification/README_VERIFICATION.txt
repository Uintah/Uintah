The verification tests are run using python's unittest framework. 

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

