this folder is for RMCRT implemented with Arches framework.

RMCRTlinearInterp.mk and RMCRTlinearInterp.cc are for RMCRT with
linear interpolation of temperature on cell faces.

RMCRTRRSD.mk and RMCRTRRSD.cc are RMCRT with RR ( Russian Rouelet )
and calculate Sandard Diviation.
the prob with SD now is, my SD is not decreasing with incresed ray
numbers.
it should be propotional to 1/ sqrt(N), N is the number of samples.


RMCRTnoInterpolation.cc and RMCRTnoInterpolation.mk are the RMCRT with
no interpolation of temperature on cell faces.
