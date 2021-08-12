
#ifndef fspec_mm_explicit_vel
#define fspec_mm_explicit_vel

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_mm_explicit_vel(int* idxLo,
                                int* idxHi,
                                int* phi_low_x, int* phi_low_y, int* phi_low_z, int* phi_high_x, int* phi_high_y, int* phi_high_z, double* phi_ptr,
                                int* old_phi_low_x, int* old_phi_low_y, int* old_phi_low_z, int* old_phi_high_x, int* old_phi_high_y, int* old_phi_high_z, double* old_phi_ptr,
                                int* ae_low_x, int* ae_low_y, int* ae_low_z, int* ae_high_x, int* ae_high_y, int* ae_high_z, double* ae_ptr,
                                int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
                                int* an_low_x, int* an_low_y, int* an_low_z, int* an_high_x, int* an_high_y, int* an_high_z, double* an_ptr,
                                int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
                                int* at_low_x, int* at_low_y, int* at_low_z, int* at_high_x, int* at_high_y, int* at_high_z, double* at_ptr,
                                int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
                                int* ap_low_x, int* ap_low_y, int* ap_low_z, int* ap_high_x, int* ap_high_y, int* ap_high_z, double* ap_ptr,
                                int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                                int* old_den_low_x, int* old_den_low_y, int* old_den_low_z, int* old_den_high_x, int* old_den_high_y, int* old_den_high_z, double* old_den_ptr,
                                int* sew_low, int* sew_high, double* sew_ptr,
                                int* sns_low, int* sns_high, double* sns_ptr,
                                int* stb_low, int* stb_high, double* stb_ptr,
                                double* dtime,
                                int* ioff,
                                int* joff,
                                int* koff,
                                int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                                int* mmwallid);

static void fort_mm_explicit_vel( Uintah::IntVector & idxLo,
                                  Uintah::IntVector & idxHi,
                                  Uintah::Array3<double> & phi,
                                  const Uintah::Array3<double> & old_phi,
                                  Uintah::Array3<double> & ae,
                                  Uintah::Array3<double> & aw,
                                  Uintah::Array3<double> & an,
                                  Uintah::Array3<double> & as,
                                  Uintah::Array3<double> & at,
                                  Uintah::Array3<double> & ab,
                                  Uintah::Array3<double> & ap,
                                  Uintah::Array3<double> & su,
                                  Uintah::constCCVariable<double> & old_den,
                                  Uintah::OffsetArray1<double> & sew,
                                  Uintah::OffsetArray1<double> & sns,
                                  Uintah::OffsetArray1<double> & stb,
                                  double & dtime,
                                  int & ioff,
                                  int & joff,
                                  int & koff,
                                  Uintah::constCCVariable<int> & pcell,
                                  int & mmwallid )
{
  Uintah::IntVector phi_low = phi.getWindow()->getOffset();
  Uintah::IntVector phi_high = phi.getWindow()->getData()->size() + phi_low - Uintah::IntVector(1, 1, 1);
  int phi_low_x = phi_low.x();
  int phi_high_x = phi_high.x();
  int phi_low_y = phi_low.y();
  int phi_high_y = phi_high.y();
  int phi_low_z = phi_low.z();
  int phi_high_z = phi_high.z();
  Uintah::IntVector old_phi_low = old_phi.getWindow()->getOffset();
  Uintah::IntVector old_phi_high = old_phi.getWindow()->getData()->size() + old_phi_low - Uintah::IntVector(1, 1, 1);
  int old_phi_low_x = old_phi_low.x();
  int old_phi_high_x = old_phi_high.x();
  int old_phi_low_y = old_phi_low.y();
  int old_phi_high_y = old_phi_high.y();
  int old_phi_low_z = old_phi_low.z();
  int old_phi_high_z = old_phi_high.z();
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
  Uintah::IntVector ap_low = ap.getWindow()->getOffset();
  Uintah::IntVector ap_high = ap.getWindow()->getData()->size() + ap_low - Uintah::IntVector(1, 1, 1);
  int ap_low_x = ap_low.x();
  int ap_high_x = ap_high.x();
  int ap_low_y = ap_low.y();
  int ap_high_y = ap_high.y();
  int ap_low_z = ap_low.z();
  int ap_high_z = ap_high.z();
  Uintah::IntVector su_low = su.getWindow()->getOffset();
  Uintah::IntVector su_high = su.getWindow()->getData()->size() + su_low - Uintah::IntVector(1, 1, 1);
  int su_low_x = su_low.x();
  int su_high_x = su_high.x();
  int su_low_y = su_low.y();
  int su_high_y = su_high.y();
  int su_low_z = su_low.z();
  int su_high_z = su_high.z();
  Uintah::IntVector old_den_low = old_den.getWindow()->getOffset();
  Uintah::IntVector old_den_high = old_den.getWindow()->getData()->size() + old_den_low - Uintah::IntVector(1, 1, 1);
  int old_den_low_x = old_den_low.x();
  int old_den_high_x = old_den_high.x();
  int old_den_low_y = old_den_low.y();
  int old_den_high_y = old_den_high.y();
  int old_den_low_z = old_den_low.z();
  int old_den_high_z = old_den_high.z();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  F_mm_explicit_vel( idxLo.get_pointer(),
                   idxHi.get_pointer(),
                   &phi_low_x, &phi_low_y, &phi_low_z, &phi_high_x, &phi_high_y, &phi_high_z, phi.getPointer(),
                   &old_phi_low_x, &old_phi_low_y, &old_phi_low_z, &old_phi_high_x, &old_phi_high_y, &old_phi_high_z, const_cast<double*>(old_phi.getPointer()),
                   &ae_low_x, &ae_low_y, &ae_low_z, &ae_high_x, &ae_high_y, &ae_high_z, ae.getPointer(),
                   &aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
                   &an_low_x, &an_low_y, &an_low_z, &an_high_x, &an_high_y, &an_high_z, an.getPointer(),
                   &as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
                   &at_low_x, &at_low_y, &at_low_z, &at_high_x, &at_high_y, &at_high_z, at.getPointer(),
                   &ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
                   &ap_low_x, &ap_low_y, &ap_low_z, &ap_high_x, &ap_high_y, &ap_high_z, ap.getPointer(),
                   &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
                   &old_den_low_x, &old_den_low_y, &old_den_low_z, &old_den_high_x, &old_den_high_y, &old_den_high_z, const_cast<double*>(old_den.getPointer()),
                   &sew_low, &sew_high, sew.get_objs(),
                   &sns_low, &sns_high, sns.get_objs(),
                   &stb_low, &stb_high, stb.get_objs(),
                   &dtime,
                   &ioff,
                   &joff,
                   &koff,
                   &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
                   &mmwallid );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine MM_EXPLICIT_VEL(idxLo, idxHi, phi_low_x, phi_low_y,
     & phi_low_z, phi_high_x, phi_high_y, phi_high_z, phi, 
     & old_phi_low_x, old_phi_low_y, old_phi_low_z, old_phi_high_x, 
     & old_phi_high_y, old_phi_high_z, old_phi, ae_low_x, ae_low_y, 
     & ae_low_z, ae_high_x, ae_high_y, ae_high_z, ae, aw_low_x, 
     & aw_low_y, aw_low_z, aw_high_x, aw_high_y, aw_high_z, aw, 
     & an_low_x, an_low_y, an_low_z, an_high_x, an_high_y, an_high_z, 
     & an, as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, 
     & as_high_z, as, at_low_x, at_low_y, at_low_z, at_high_x, 
     & at_high_y, at_high_z, at, ab_low_x, ab_low_y, ab_low_z, 
     & ab_high_x, ab_high_y, ab_high_z, ab, ap_low_x, ap_low_y, 
     & ap_low_z, ap_high_x, ap_high_y, ap_high_z, ap, su_low_x, 
     & su_low_y, su_low_z, su_high_x, su_high_y, su_high_z, su, 
     & old_den_low_x, old_den_low_y, old_den_low_z, old_den_high_x, 
     & old_den_high_y, old_den_high_z, old_den, sew_low, sew_high, sew,
     &  sns_low, sns_high, sns, stb_low, stb_high, stb, dtime, ioff, 
     & joff, koff, pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x,
     &  pcell_high_y, pcell_high_z, pcell, mmwallid)

      implicit none
      integer idxLo(3)
      integer idxHi(3)
      integer phi_low_x, phi_low_y, phi_low_z, phi_high_x, phi_high_y, 
     & phi_high_z
      double precision phi(phi_low_x:phi_high_x, phi_low_y:phi_high_y, 
     & phi_low_z:phi_high_z)
      integer old_phi_low_x, old_phi_low_y, old_phi_low_z, 
     & old_phi_high_x, old_phi_high_y, old_phi_high_z
      double precision old_phi(old_phi_low_x:old_phi_high_x, 
     & old_phi_low_y:old_phi_high_y, old_phi_low_z:old_phi_high_z)
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
      integer ap_low_x, ap_low_y, ap_low_z, ap_high_x, ap_high_y, 
     & ap_high_z
      double precision ap(ap_low_x:ap_high_x, ap_low_y:ap_high_y, 
     & ap_low_z:ap_high_z)
      integer su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z
      double precision su(su_low_x:su_high_x, su_low_y:su_high_y, 
     & su_low_z:su_high_z)
      integer old_den_low_x, old_den_low_y, old_den_low_z, 
     & old_den_high_x, old_den_high_y, old_den_high_z
      double precision old_den(old_den_low_x:old_den_high_x, 
     & old_den_low_y:old_den_high_y, old_den_low_z:old_den_high_z)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      double precision dtime
      integer ioff
      integer joff
      integer koff
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer mmwallid
#endif /* __cplusplus */

#endif /* fspec_mm_explicit_vel */
