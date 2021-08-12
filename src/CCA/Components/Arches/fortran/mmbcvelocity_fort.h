
#ifndef fspec_mmbcvelocity
#define fspec_mmbcvelocity

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_mmbcvelocity(int* idxLoU,
                              int* idxHiU,
                              int* ae_low_x, int* ae_low_y, int* ae_low_z, int* ae_high_x, int* ae_high_y, int* ae_high_z, double* ae_ptr,
                              int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
                              int* an_low_x, int* an_low_y, int* an_low_z, int* an_high_x, int* an_high_y, int* an_high_z, double* an_ptr,
                              int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
                              int* at_low_x, int* at_low_y, int* at_low_z, int* at_high_x, int* at_high_y, int* at_high_z, double* at_ptr,
                              int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
                              int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                              int* sp_low_x, int* sp_low_y, int* sp_low_z, int* sp_high_x, int* sp_high_y, int* sp_high_z, double* sp_ptr,
                              int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                              int* mmwallid,
                              int* ioff,
                              int* joff,
                              int* koff);

static void fort_mmbcvelocity( Uintah::IntVector & idxLoU,
                               Uintah::IntVector & idxHiU,
                               Uintah::Array3<double> & ae,
                               Uintah::Array3<double> & aw,
                               Uintah::Array3<double> & an,
                               Uintah::Array3<double> & as,
                               Uintah::Array3<double> & at,
                               Uintah::Array3<double> & ab,
                               Uintah::Array3<double> & su,
                               Uintah::Array3<double> & sp,
                               Uintah::constCCVariable<int> & pcell,
                               int & mmwallid,
                               int & ioff,
                               int & joff,
                               int & koff )
{
  Uintah::IntVector ae_low = ae.getWindow()->getOffset();
  Uintah::IntVector ae_high = ae.getWindow()->getData()->size() + ae_low - Uintah::IntVector(1, 1, 1);
  int ae_low_x = ae_low.x();
  int ae_high_x = ae_high.x();
  int ae_low_y = ae_low.y();
  int ae_high_y = ae_high.y();
  int ae_low_z = ae_low.z();
  int ae_high_z = ae_high.z();
  Uintah::IntVector aw_low = aw.getWindow()->getOffset();
  Uintah::IntVector aw_high = aw.getWindow()->getData()->size() + aw_low - Uintah::IntVector(1, 1, 1);
  int aw_low_x = aw_low.x();
  int aw_high_x = aw_high.x();
  int aw_low_y = aw_low.y();
  int aw_high_y = aw_high.y();
  int aw_low_z = aw_low.z();
  int aw_high_z = aw_high.z();
  Uintah::IntVector an_low = an.getWindow()->getOffset();
  Uintah::IntVector an_high = an.getWindow()->getData()->size() + an_low - Uintah::IntVector(1, 1, 1);
  int an_low_x = an_low.x();
  int an_high_x = an_high.x();
  int an_low_y = an_low.y();
  int an_high_y = an_high.y();
  int an_low_z = an_low.z();
  int an_high_z = an_high.z();
  Uintah::IntVector as_low = as.getWindow()->getOffset();
  Uintah::IntVector as_high = as.getWindow()->getData()->size() + as_low - Uintah::IntVector(1, 1, 1);
  int as_low_x = as_low.x();
  int as_high_x = as_high.x();
  int as_low_y = as_low.y();
  int as_high_y = as_high.y();
  int as_low_z = as_low.z();
  int as_high_z = as_high.z();
  Uintah::IntVector at_low = at.getWindow()->getOffset();
  Uintah::IntVector at_high = at.getWindow()->getData()->size() + at_low - Uintah::IntVector(1, 1, 1);
  int at_low_x = at_low.x();
  int at_high_x = at_high.x();
  int at_low_y = at_low.y();
  int at_high_y = at_high.y();
  int at_low_z = at_low.z();
  int at_high_z = at_high.z();
  Uintah::IntVector ab_low = ab.getWindow()->getOffset();
  Uintah::IntVector ab_high = ab.getWindow()->getData()->size() + ab_low - Uintah::IntVector(1, 1, 1);
  int ab_low_x = ab_low.x();
  int ab_high_x = ab_high.x();
  int ab_low_y = ab_low.y();
  int ab_high_y = ab_high.y();
  int ab_low_z = ab_low.z();
  int ab_high_z = ab_high.z();
  Uintah::IntVector su_low = su.getWindow()->getOffset();
  Uintah::IntVector su_high = su.getWindow()->getData()->size() + su_low - Uintah::IntVector(1, 1, 1);
  int su_low_x = su_low.x();
  int su_high_x = su_high.x();
  int su_low_y = su_low.y();
  int su_high_y = su_high.y();
  int su_low_z = su_low.z();
  int su_high_z = su_high.z();
  Uintah::IntVector sp_low = sp.getWindow()->getOffset();
  Uintah::IntVector sp_high = sp.getWindow()->getData()->size() + sp_low - Uintah::IntVector(1, 1, 1);
  int sp_low_x = sp_low.x();
  int sp_high_x = sp_high.x();
  int sp_low_y = sp_low.y();
  int sp_high_y = sp_high.y();
  int sp_low_z = sp_low.z();
  int sp_high_z = sp_high.z();
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  F_mmbcvelocity( idxLoU.get_pointer(),
                 idxHiU.get_pointer(),
                 &ae_low_x, &ae_low_y, &ae_low_z, &ae_high_x, &ae_high_y, &ae_high_z, ae.getPointer(),
                 &aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
                 &an_low_x, &an_low_y, &an_low_z, &an_high_x, &an_high_y, &an_high_z, an.getPointer(),
                 &as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
                 &at_low_x, &at_low_y, &at_low_z, &at_high_x, &at_high_y, &at_high_z, at.getPointer(),
                 &ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
                 &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
                 &sp_low_x, &sp_low_y, &sp_low_z, &sp_high_x, &sp_high_y, &sp_high_z, sp.getPointer(),
                 &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
                 &mmwallid,
                 &ioff,
                 &joff,
                 &koff );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine mmbcvelocity(idxLoU, idxHiU, ae_low_x, ae_low_y, 
     & ae_low_z, ae_high_x, ae_high_y, ae_high_z, ae, aw_low_x, 
     & aw_low_y, aw_low_z, aw_high_x, aw_high_y, aw_high_z, aw, 
     & an_low_x, an_low_y, an_low_z, an_high_x, an_high_y, an_high_z, 
     & an, as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, 
     & as_high_z, as, at_low_x, at_low_y, at_low_z, at_high_x, 
     & at_high_y, at_high_z, at, ab_low_x, ab_low_y, ab_low_z, 
     & ab_high_x, ab_high_y, ab_high_z, ab, su_low_x, su_low_y, 
     & su_low_z, su_high_x, su_high_y, su_high_z, su, sp_low_x, 
     & sp_low_y, sp_low_z, sp_high_x, sp_high_y, sp_high_z, sp, 
     & pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z, pcell, mmwallid, ioff, joff, koff)

      implicit none
      integer idxLoU(3)
      integer idxHiU(3)
      integer ae_low_x, ae_low_y, ae_low_z, ae_high_x, ae_high_y, 
     & ae_high_z
      double precision ae(ae_low_x:ae_high_x, ae_low_y:ae_high_y, 
     & ae_low_z:ae_high_z)
      integer aw_low_x, aw_low_y, aw_low_z, aw_high_x, aw_high_y, 
     & aw_high_z
      double precision aw(aw_low_x:aw_high_x, aw_low_y:aw_high_y, 
     & aw_low_z:aw_high_z)
      integer an_low_x, an_low_y, an_low_z, an_high_x, an_high_y, 
     & an_high_z
      double precision an(an_low_x:an_high_x, an_low_y:an_high_y, 
     & an_low_z:an_high_z)
      integer as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, 
     & as_high_z
      double precision as(as_low_x:as_high_x, as_low_y:as_high_y, 
     & as_low_z:as_high_z)
      integer at_low_x, at_low_y, at_low_z, at_high_x, at_high_y, 
     & at_high_z
      double precision at(at_low_x:at_high_x, at_low_y:at_high_y, 
     & at_low_z:at_high_z)
      integer ab_low_x, ab_low_y, ab_low_z, ab_high_x, ab_high_y, 
     & ab_high_z
      double precision ab(ab_low_x:ab_high_x, ab_low_y:ab_high_y, 
     & ab_low_z:ab_high_z)
      integer su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z
      double precision su(su_low_x:su_high_x, su_low_y:su_high_y, 
     & su_low_z:su_high_z)
      integer sp_low_x, sp_low_y, sp_low_z, sp_high_x, sp_high_y, 
     & sp_high_z
      double precision sp(sp_low_x:sp_high_x, sp_low_y:sp_high_y, 
     & sp_low_z:sp_high_z)
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer mmwallid
      integer ioff
      integer joff
      integer koff
#endif /* __cplusplus */

#endif /* fspec_mmbcvelocity */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
