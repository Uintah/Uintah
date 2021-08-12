
#ifndef fspec_rqpart
#define fspec_rqpart

#include <CCA/Components/Arches/CoalModels/fortran/FortranNameMangle.h>

#ifdef __cplusplus

extern "C" void RQPART(double* dia,
                       double* tmp,
                       double* omegaa,
                       double* afrac0,
                       double* qabs,
                       double* qsca);

static void fort_rqpart( double & dia,
                         double & tmp,
                         double & omegaa,
                         double & afrac0,
                         double & qabs,
                         double & qsca )
{
  RQPART( &dia,
          &tmp,
          &omegaa,
          &afrac0,
          &qabs,
          &qsca );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine rqpart(dia, tmp, omegaa, afrac0, qabs, qsca)

      implicit none
      double precision dia
      double precision tmp
      double precision omegaa
      double precision afrac0
      double precision qabs
      double precision qsca
#endif /* __cplusplus */

#endif /* fspec_rqpart */

