
#ifndef fspec_rordrtn
#define fspec_rordrtn

#ifdef __cplusplus

#include <CCA/Components/Arches/Radiation/fortran/FortranNameMangle.h>

extern "C" void F_rordrtn(int* sn,
                        int* ord_low, int* ord_high, double* ord_ptr,
                        int* oxi_low, int* oxi_high, double* oxi_ptr,
                        int* omu_low, int* omu_high, double* omu_ptr,
                        int* oeta_low, int* oeta_high, double* oeta_ptr,
                        int* wt_low, int* wt_high, double* wt_ptr);

static void fort_rordrtn( int & sn,
                          Uintah::OffsetArray1<double> & ord,
                          Uintah::OffsetArray1<double> & oxi,
                          Uintah::OffsetArray1<double> & omu,
                          Uintah::OffsetArray1<double> & oeta,
                          Uintah::OffsetArray1<double> & wt )
{
  int ord_low = ord.low();
  int ord_high = ord.high();
  int oxi_low = oxi.low();
  int oxi_high = oxi.high();
  int omu_low = omu.low();
  int omu_high = omu.high();
  int oeta_low = oeta.low();
  int oeta_high = oeta.high();
  int wt_low = wt.low();
  int wt_high = wt.high();
  F_rordrtn( &sn,
           &ord_low, &ord_high, ord.get_objs(),
           &oxi_low, &oxi_high, oxi.get_objs(),
           &omu_low, &omu_high, omu.get_objs(),
           &oeta_low, &oeta_high, oeta.get_objs(),
           &wt_low, &wt_high, wt.get_objs() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine RORDRTN(sn, ord_low, ord_high, ord, oxi_low, oxi_high,
     &  oxi, omu_low, omu_high, omu, oeta_low, oeta_high, oeta, wt_low,
     &  wt_high, wt)

      implicit none
      integer sn
      integer ord_low
      integer ord_high
      double precision ord(ord_low:ord_high)
      integer oxi_low
      integer oxi_high
      double precision oxi(oxi_low:oxi_high)
      integer omu_low
      integer omu_high
      double precision omu(omu_low:omu_high)
      integer oeta_low
      integer oeta_high
      double precision oeta(oeta_low:oeta_high)
      integer wt_low
      integer wt_high
      double precision wt(wt_low:wt_high)
#endif /* __cplusplus */

#endif /* fspec_rordrtn */
