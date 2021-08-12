
#ifndef fspec_pressrcpred_var
#define fspec_pressrcpred_var

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_pressrcpred_var(int* idxLo,
                                int* idxHi,
                                int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                                int* divU_low_x, int* divU_low_y, int* divU_low_z, int* divU_high_x, int* divU_high_y, int* divU_high_z, double* divU_ptr,
                                int* uhat_low_x, int* uhat_low_y, int* uhat_low_z, int* uhat_high_x, int* uhat_high_y, int* uhat_high_z, double* uhat_ptr,
                                int* vhat_low_x, int* vhat_low_y, int* vhat_low_z, int* vhat_high_x, int* vhat_high_y, int* vhat_high_z, double* vhat_ptr,
                                int* what_low_x, int* what_low_y, int* what_low_z, int* what_high_x, int* what_high_y, int* what_high_z, double* what_ptr,
                                double* delta_t,
                                int* sew_low, int* sew_high, double* sew_ptr,
                                int* sns_low, int* sns_high, double* sns_ptr,
                                int* stb_low, int* stb_high, double* stb_ptr);

static void fort_pressrcpred_var( Uintah::IntVector & idxLo,
                                  Uintah::IntVector & idxHi,
                                  Uintah::CCVariable<double> & su,
                                  Uintah::constCCVariable<double> & divU,
                                  Uintah::constSFCXVariable<double> & uhat,
                                  Uintah::constSFCYVariable<double> & vhat,
                                  Uintah::constSFCZVariable<double> & what,
                                  double & delta_t,
                                  Uintah::OffsetArray1<double> & sew,
                                  Uintah::OffsetArray1<double> & sns,
                                  Uintah::OffsetArray1<double> & stb )
{
  Uintah::IntVector su_low = su.getWindow()->getOffset();
  Uintah::IntVector su_high = su.getWindow()->getData()->size() + su_low - Uintah::IntVector(1, 1, 1);
  int su_low_x = su_low.x();
  int su_high_x = su_high.x();
  int su_low_y = su_low.y();
  int su_high_y = su_high.y();
  int su_low_z = su_low.z();
  int su_high_z = su_high.z();
  Uintah::IntVector divU_low = divU.getWindow()->getOffset();
  Uintah::IntVector divU_high = divU.getWindow()->getData()->size() + divU_low - Uintah::IntVector(1, 1, 1);
  int divU_low_x = divU_low.x();
  int divU_high_x = divU_high.x();
  int divU_low_y = divU_low.y();
  int divU_high_y = divU_high.y();
  int divU_low_z = divU_low.z();
  int divU_high_z = divU_high.z();
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
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  F_pressrcpred_var( idxLo.get_pointer(),
                   idxHi.get_pointer(),
                   &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
                   &divU_low_x, &divU_low_y, &divU_low_z, &divU_high_x, &divU_high_y, &divU_high_z, const_cast<double*>(divU.getPointer()),
                   &uhat_low_x, &uhat_low_y, &uhat_low_z, &uhat_high_x, &uhat_high_y, &uhat_high_z, const_cast<double*>(uhat.getPointer()),
                   &vhat_low_x, &vhat_low_y, &vhat_low_z, &vhat_high_x, &vhat_high_y, &vhat_high_z, const_cast<double*>(vhat.getPointer()),
                   &what_low_x, &what_low_y, &what_low_z, &what_high_x, &what_high_y, &what_high_z, const_cast<double*>(what.getPointer()),
                   &delta_t,
                   &sew_low, &sew_high, sew.get_objs(),
                   &sns_low, &sns_high, sns.get_objs(),
                   &stb_low, &stb_high, stb.get_objs() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine PRESSRCPRED_VAR(idxLo, idxHi, su_low_x, su_low_y,
     & su_low_z, su_high_x, su_high_y, su_high_z, su, divU_low_x, 
     & divU_low_y, divU_low_z, divU_high_x, divU_high_y, divU_high_z, 
     & divU, uhat_low_x, uhat_low_y, uhat_low_z, uhat_high_x, 
     & uhat_high_y, uhat_high_z, uhat, vhat_low_x, vhat_low_y, 
     & vhat_low_z, vhat_high_x, vhat_high_y, vhat_high_z, vhat, 
     & what_low_x, what_low_y, what_low_z, what_high_x, what_high_y, 
     & what_high_z, what, delta_t, sew_low, sew_high, sew, sns_low, 
     & sns_high, sns, stb_low, stb_high, stb)

      implicit none
      integer idxLo(3)
      integer idxHi(3)
      integer su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z
      double precision su(su_low_x:su_high_x, su_low_y:su_high_y, 
     & su_low_z:su_high_z)
      integer divU_low_x, divU_low_y, divU_low_z, divU_high_x, 
     & divU_high_y, divU_high_z
      double precision divU(divU_low_x:divU_high_x, divU_low_y:
     & divU_high_y, divU_low_z:divU_high_z)
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
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
#endif /* __cplusplus */

#endif /* fspec_pressrcpred_var */
