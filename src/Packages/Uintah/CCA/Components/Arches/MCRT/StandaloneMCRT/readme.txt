----------------------------------
MCRToriginal 

is the original with Russian Roulette. 
the old scheme of scattering ( if scat_len < length , scatter,
otherwise no)
the old scheme of scattering will "fail" on finer mesh, as there will
be fewer and fewer chance of scat_len < length, as length decreases
with finer mesh.

The main .cc drivers are
RMCRT.cc ( a incomplete driver )

RMCRTnoInterpolation.cc ( with RR, but no linear interpolation for T
on cell faces)

RMCRTtemplatewithRR.cc ( with RR, and with linear interpolation for T
and a on cell faces)

RMCRTcellq.cc ( with RR, to calculate heat flux on faces, not done
yet)

ray.cc, ray.h, VirtualSurface.cc, VirtualSurface.h, RealSurface.cc ,
RealSurface.h are different from the ones in other folders.

----------------------------------------------

MCRTscatter1ver and MCRTscatterlen 

are folders with different modified scattering schemes.

MCRTscatter1ver ( first version) has more potential. it is pick a scat_len, 
then check scat_len < pre_straight_len , and ( scat_len <=
straight_len && scat_len >= pre_straight_len) scattering happens, 
otherwise, go straight.
it is changed in ray.cc , TravelinMediumInten function.

MCRTscatterlen, set an arbitrary length to compare with scat_len, 
which dramatically affects the final results.

---------------------------------------------
MCRTStratified

is use stratification sampling.
RMCRTRRSD.cc doesnot use stratified sampling.
RMCRTRRSDStratified.cc used stratified sampling.

ray.cc has been changed here. for R_phi, R_theta.
------------------------
