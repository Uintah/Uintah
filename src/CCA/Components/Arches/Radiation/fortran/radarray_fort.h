
#ifndef fspec_radarray
#define fspec_radarray

#ifdef __cplusplus

#include <CCA/Components/Arches/Radiation/fortran/FortranNameMangle.h>

extern "C" void F_radarray(int* rgamma_low, int* rgamma_high, double* rgamma_ptr,
                         int* sd15_low, int* sd15_high, double* sd15_ptr,
                         int* sd_low, int* sd_high, double* sd_ptr,
                         int* sd7_low, int* sd7_high, double* sd7_ptr,
                         int* sd3_low, int* sd3_high, double* sd3_ptr);

static void fort_radarray( Uintah::OffsetArray1<double> & rgamma,
                           Uintah::OffsetArray1<double> & sd15,
                           Uintah::OffsetArray1<double> & sd,
                           Uintah::OffsetArray1<double> & sd7,
                           Uintah::OffsetArray1<double> & sd3 )
{
  int rgamma_low = rgamma.low();
  int rgamma_high = rgamma.high();
  int sd15_low = sd15.low();
  int sd15_high = sd15.high();
  int sd_low = sd.low();
  int sd_high = sd.high();
  int sd7_low = sd7.low();
  int sd7_high = sd7.high();
  int sd3_low = sd3.low();
  int sd3_high = sd3.high();
  F_radarray( &rgamma_low, &rgamma_high, rgamma.get_objs(),
            &sd15_low, &sd15_high, sd15.get_objs(),
            &sd_low, &sd_high, sd.get_objs(),
            &sd7_low, &sd7_high, sd7.get_objs(),
            &sd3_low, &sd3_high, sd3.get_objs() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine RADARRAY(rgamma_low, rgamma_high, rgamma, sd15_low,
     & sd15_high, sd15, sd_low, sd_high, sd, sd7_low, sd7_high, sd7, 
     & sd3_low, sd3_high, sd3)

      implicit none
      integer rgamma_low
      integer rgamma_high
      double precision rgamma(rgamma_low:rgamma_high)
      integer sd15_low
      integer sd15_high
      double precision sd15(sd15_low:sd15_high)
      integer sd_low
      integer sd_high
      double precision sd(sd_low:sd_high)
      integer sd7_low
      integer sd7_high
      double precision sd7(sd7_low:sd7_high)
      integer sd3_low
      integer sd3_high
      double precision sd3(sd3_low:sd3_high)
#endif /* __cplusplus */

#endif /* fspec_radarray */
