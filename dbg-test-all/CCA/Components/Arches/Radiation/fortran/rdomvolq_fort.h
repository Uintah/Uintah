
#ifndef fspec_rdomvolq
#define fspec_rdomvolq

#ifdef __cplusplus

extern "C" void rdomvolq_(int* idxlo,
                          int* idxhi,
                          int* l,
                          int* wt_low, int* wt_high, double* wt_ptr,
                          int* cenint_low_x, int* cenint_low_y, int* cenint_low_z, int* cenint_high_x, int* cenint_high_y, int* cenint_high_z, double* cenint_ptr,
                          int* volq_low_x, int* volq_low_y, int* volq_low_z, int* volq_high_x, int* volq_high_y, int* volq_high_z, double* volq_ptr);

static void fort_rdomvolq( Uintah::IntVector & idxlo,
                           Uintah::IntVector & idxhi,
                           int & l,
                           Uintah::OffsetArray1<double> & wt,
                           Uintah::CCVariable<double> & cenint,
                           Uintah::CCVariable<double> & volq )
{
  int wt_low = wt.low();
  int wt_high = wt.high();
  Uintah::IntVector cenint_low = cenint.getWindow()->getOffset();
  Uintah::IntVector cenint_high = cenint.getWindow()->getData()->size() + cenint_low - Uintah::IntVector(1, 1, 1);
  int cenint_low_x = cenint_low.x();
  int cenint_high_x = cenint_high.x();
  int cenint_low_y = cenint_low.y();
  int cenint_high_y = cenint_high.y();
  int cenint_low_z = cenint_low.z();
  int cenint_high_z = cenint_high.z();
  Uintah::IntVector volq_low = volq.getWindow()->getOffset();
  Uintah::IntVector volq_high = volq.getWindow()->getData()->size() + volq_low - Uintah::IntVector(1, 1, 1);
  int volq_low_x = volq_low.x();
  int volq_high_x = volq_high.x();
  int volq_low_y = volq_low.y();
  int volq_high_y = volq_high.y();
  int volq_low_z = volq_low.z();
  int volq_high_z = volq_high.z();
  rdomvolq_( idxlo.get_pointer(),
             idxhi.get_pointer(),
             &l,
             &wt_low, &wt_high, wt.get_objs(),
             &cenint_low_x, &cenint_low_y, &cenint_low_z, &cenint_high_x, &cenint_high_y, &cenint_high_z, cenint.getPointer(),
             &volq_low_x, &volq_low_y, &volq_low_z, &volq_high_x, &volq_high_y, &volq_high_z, volq.getPointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine rdomvolq(idxlo, idxhi, l, wt_low, wt_high, wt, 
     & cenint_low_x, cenint_low_y, cenint_low_z, cenint_high_x, 
     & cenint_high_y, cenint_high_z, cenint, volq_low_x, volq_low_y, 
     & volq_low_z, volq_high_x, volq_high_y, volq_high_z, volq)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
      integer l
      integer wt_low
      integer wt_high
      double precision wt(wt_low:wt_high)
      integer cenint_low_x, cenint_low_y, cenint_low_z, cenint_high_x, 
     & cenint_high_y, cenint_high_z
      double precision cenint(cenint_low_x:cenint_high_x, cenint_low_y:
     & cenint_high_y, cenint_low_z:cenint_high_z)
      integer volq_low_x, volq_low_y, volq_low_z, volq_high_x, 
     & volq_high_y, volq_high_z
      double precision volq(volq_low_x:volq_high_x, volq_low_y:
     & volq_high_y, volq_low_z:volq_high_z)
#endif /* __cplusplus */

#endif /* fspec_rdomvolq */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
