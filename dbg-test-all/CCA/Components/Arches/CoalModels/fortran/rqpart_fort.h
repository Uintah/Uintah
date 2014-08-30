
#ifndef fspec_rqpart
#define fspec_rqpart

#ifdef __cplusplus

extern "C" void rqpart_(double* dia,
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
  rqpart_( &dia,
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

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
