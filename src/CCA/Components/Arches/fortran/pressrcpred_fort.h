
#ifndef fspec_pressrcpred
#define fspec_pressrcpred

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_pressrcpred(int* idxLo,
                            int* idxHi,
                            int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                            int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                            int* uhat_low_x, int* uhat_low_y, int* uhat_low_z, int* uhat_high_x, int* uhat_high_y, int* uhat_high_z, double* uhat_ptr,
                            int* vhat_low_x, int* vhat_low_y, int* vhat_low_z, int* vhat_high_x, int* vhat_high_y, int* vhat_high_z, double* vhat_ptr,
                            int* what_low_x, int* what_low_y, int* what_low_z, int* what_high_x, int* what_high_y, int* what_high_z, double* what_ptr,
                            double* delta_t,
                            double* dx,
                            double* dy,
                            double* dz);

static void fort_pressrcpred( Uintah::IntVector & idxLo,
                              Uintah::IntVector & idxHi,
                              Uintah::CCVariable<double> & su,
                              Uintah::constCCVariable<double> & den,
                              Uintah::constSFCXVariable<double> & uhat,
                              Uintah::constSFCYVariable<double> & vhat,
                              Uintah::constSFCZVariable<double> & what,
                              double & delta_t,
                              double & dx,
                              double & dy,
                              double & dz )
{
  Uintah::IntVector su_low = su.getWindow()->getOffset();
  Uintah::IntVector su_high = su.getWindow()->getData()->size() + su_low - Uintah::IntVector(1, 1, 1);
  int su_low_x = su_low.x();
  int su_high_x = su_high.x();
  int su_low_y = su_low.y();
  int su_high_y = su_high.y();
  int su_low_z = su_low.z();
  int su_high_z = su_high.z();
  Uintah::IntVector den_low = den.getWindow()->getOffset();
  Uintah::IntVector den_high = den.getWindow()->getData()->size() + den_low - Uintah::IntVector(1, 1, 1);
  int den_low_x = den_low.x();
  int den_high_x = den_high.x();
  int den_low_y = den_low.y();
  int den_high_y = den_high.y();
  int den_low_z = den_low.z();
  int den_high_z = den_high.z();
  Uintah::IntVector uhat_low = uhat.getWindow()->getOffset();
  Uintah::IntVector uhat_high = uhat.getWindow()->getData()->size() + uhat_low - Uintah::IntVector(1, 1, 1);
  int uhat_low_x = uhat_low.x();
  int uhat_high_x = uhat_high.x();
  int uhat_low_y = uhat_low.y();
  int uhat_high_y = uhat_high.y();
  int uhat_low_z = uhat_low.z();
  int uhat_high_z = uhat_high.z();
  Uintah::IntVector vhat_low = vhat.getWindow()->getOffset();
  Uintah::IntVector vhat_high = vhat.getWindow()->getData()->size() + vhat_low - Uintah::IntVector(1, 1, 1);
  int vhat_low_x = vhat_low.x();
  int vhat_high_x = vhat_high.x();
  int vhat_low_y = vhat_low.y();
  int vhat_high_y = vhat_high.y();
  int vhat_low_z = vhat_low.z();
  int vhat_high_z = vhat_high.z();
  Uintah::IntVector what_low = what.getWindow()->getOffset();
  Uintah::IntVector what_high = what.getWindow()->getData()->size() + what_low - Uintah::IntVector(1, 1, 1);
  int what_low_x = what_low.x();
  int what_high_x = what_high.x();
  int what_low_y = what_low.y();
  int what_high_y = what_high.y();
  int what_low_z = what_low.z();
  int what_high_z = what_high.z();
  F_pressrcpred( idxLo.get_pointer(),
               idxHi.get_pointer(),
               &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
               &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
               &uhat_low_x, &uhat_low_y, &uhat_low_z, &uhat_high_x, &uhat_high_y, &uhat_high_z, const_cast<double*>(uhat.getPointer()),
               &vhat_low_x, &vhat_low_y, &vhat_low_z, &vhat_high_x, &vhat_high_y, &vhat_high_z, const_cast<double*>(vhat.getPointer()),
               &what_low_x, &what_low_y, &what_low_z, &what_high_x, &what_high_y, &what_high_z, const_cast<double*>(what.getPointer()),
               &delta_t,
               &dx,
               &dy,
               &dz );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine PRESSRCPRED(idxLo, idxHi, su_low_x, su_low_y, su_low_z
     & , su_high_x, su_high_y, su_high_z, su, den_low_x, den_low_y, 
     & den_low_z, den_high_x, den_high_y, den_high_z, den, uhat_low_x, 
     & uhat_low_y, uhat_low_z, uhat_high_x, uhat_high_y, uhat_high_z, 
     & uhat, vhat_low_x, vhat_low_y, vhat_low_z, vhat_high_x, 
     & vhat_high_y, vhat_high_z, vhat, what_low_x, what_low_y, 
     & what_low_z, what_high_x, what_high_y, what_high_z, what, delta_t
     & , dx, dy, dz)

      implicit none
      integer idxLo(3)
      integer idxHi(3)
      integer su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z
      double precision su(su_low_x:su_high_x, su_low_y:su_high_y, 
     & su_low_z:su_high_z)
      integer den_low_x, den_low_y, den_low_z, den_high_x, den_high_y, 
     & den_high_z
      double precision den(den_low_x:den_high_x, den_low_y:den_high_y, 
     & den_low_z:den_high_z)
      integer uhat_low_x, uhat_low_y, uhat_low_z, uhat_high_x, 
     & uhat_high_y, uhat_high_z
      double precision uhat(uhat_low_x:uhat_high_x, uhat_low_y:
     & uhat_high_y, uhat_low_z:uhat_high_z)
      integer vhat_low_x, vhat_low_y, vhat_low_z, vhat_high_x, 
     & vhat_high_y, vhat_high_z
      double precision vhat(vhat_low_x:vhat_high_x, vhat_low_y:
     & vhat_high_y, vhat_low_z:vhat_high_z)
      integer what_low_x, what_low_y, what_low_z, what_high_x, 
     & what_high_y, what_high_z
      double precision what(what_low_x:what_high_x, what_low_y:
     & what_high_y, what_low_z:what_high_z)
      double precision delta_t
      double precision dx
      double precision dy
      double precision dz
#endif /* __cplusplus */

#endif /* fspec_pressrcpred */
