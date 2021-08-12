
#ifndef fspec_prescoef_var
#define fspec_prescoef_var

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_prescoef_var(int* idxLo,
                             int* idxHi,
                             int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                             int* ae_low_x, int* ae_low_y, int* ae_low_z, int* ae_high_x, int* ae_high_y, int* ae_high_z, double* ae_ptr,
                             int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
                             int* an_low_x, int* an_low_y, int* an_low_z, int* an_high_x, int* an_high_y, int* an_high_z, double* an_ptr,
                             int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
                             int* at_low_x, int* at_low_y, int* at_low_z, int* at_high_x, int* at_high_y, int* at_high_z, double* at_ptr,
                             int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
                             int* sew_low, int* sew_high, double* sew_ptr,
                             int* sns_low, int* sns_high, double* sns_ptr,
                             int* stb_low, int* stb_high, double* stb_ptr,
                             int* sewu_low, int* sewu_high, double* sewu_ptr,
                             int* dxep_low, int* dxep_high, double* dxep_ptr,
                             int* dxpw_low, int* dxpw_high, double* dxpw_ptr,
                             int* snsv_low, int* snsv_high, double* snsv_ptr,
                             int* dynp_low, int* dynp_high, double* dynp_ptr,
                             int* dyps_low, int* dyps_high, double* dyps_ptr,
                             int* stbw_low, int* stbw_high, double* stbw_ptr,
                             int* dztp_low, int* dztp_high, double* dztp_ptr,
                             int* dzpb_low, int* dzpb_high, double* dzpb_ptr);

static void fort_prescoef_var( Uintah::IntVector & idxLo,
                               Uintah::IntVector & idxHi,
                               Uintah::constCCVariable<double> & den,
                               Uintah::CCVariable<double> & ae,
                               Uintah::CCVariable<double> & aw,
                               Uintah::CCVariable<double> & an,
                               Uintah::CCVariable<double> & as,
                               Uintah::CCVariable<double> & at,
                               Uintah::CCVariable<double> & ab,
                               Uintah::OffsetArray1<double> & sew,
                               Uintah::OffsetArray1<double> & sns,
                               Uintah::OffsetArray1<double> & stb,
                               Uintah::OffsetArray1<double> & sewu,
                               Uintah::OffsetArray1<double> & dxep,
                               Uintah::OffsetArray1<double> & dxpw,
                               Uintah::OffsetArray1<double> & snsv,
                               Uintah::OffsetArray1<double> & dynp,
                               Uintah::OffsetArray1<double> & dyps,
                               Uintah::OffsetArray1<double> & stbw,
                               Uintah::OffsetArray1<double> & dztp,
                               Uintah::OffsetArray1<double> & dzpb )
{
  Uintah::IntVector den_low = den.getWindow()->getOffset();
  Uintah::IntVector den_high = den.getWindow()->getData()->size() + den_low - Uintah::IntVector(1, 1, 1);
  int den_low_x = den_low.x();
  int den_high_x = den_high.x();
  int den_low_y = den_low.y();
  int den_high_y = den_high.y();
  int den_low_z = den_low.z();
  int den_high_z = den_high.z();
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
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  int sewu_low = sewu.low();
  int sewu_high = sewu.high();
  int dxep_low = dxep.low();
  int dxep_high = dxep.high();
  int dxpw_low = dxpw.low();
  int dxpw_high = dxpw.high();
  int snsv_low = snsv.low();
  int snsv_high = snsv.high();
  int dynp_low = dynp.low();
  int dynp_high = dynp.high();
  int dyps_low = dyps.low();
  int dyps_high = dyps.high();
  int stbw_low = stbw.low();
  int stbw_high = stbw.high();
  int dztp_low = dztp.low();
  int dztp_high = dztp.high();
  int dzpb_low = dzpb.low();
  int dzpb_high = dzpb.high();
  F_prescoef_var( idxLo.get_pointer(),
                idxHi.get_pointer(),
                &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
                &ae_low_x, &ae_low_y, &ae_low_z, &ae_high_x, &ae_high_y, &ae_high_z, ae.getPointer(),
                &aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
                &an_low_x, &an_low_y, &an_low_z, &an_high_x, &an_high_y, &an_high_z, an.getPointer(),
                &as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
                &at_low_x, &at_low_y, &at_low_z, &at_high_x, &at_high_y, &at_high_z, at.getPointer(),
                &ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
                &sew_low, &sew_high, sew.get_objs(),
                &sns_low, &sns_high, sns.get_objs(),
                &stb_low, &stb_high, stb.get_objs(),
                &sewu_low, &sewu_high, sewu.get_objs(),
                &dxep_low, &dxep_high, dxep.get_objs(),
                &dxpw_low, &dxpw_high, dxpw.get_objs(),
                &snsv_low, &snsv_high, snsv.get_objs(),
                &dynp_low, &dynp_high, dynp.get_objs(),
                &dyps_low, &dyps_high, dyps.get_objs(),
                &stbw_low, &stbw_high, stbw.get_objs(),
                &dztp_low, &dztp_high, dztp.get_objs(),
                &dzpb_low, &dzpb_high, dzpb.get_objs() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine prescoef_var(idxLo, idxHi, den_low_x, den_low_y,
     & den_low_z, den_high_x, den_high_y, den_high_z, den, ae_low_x,
     & ae_low_y, ae_low_z, ae_high_x, ae_high_y, ae_high_z, ae,
     & aw_low_x, aw_low_y, aw_low_z, aw_high_x, aw_high_y, aw_high_z,
     & aw, an_low_x, an_low_y, an_low_z, an_high_x, an_high_y,
     & an_high_z, an, as_low_x, as_low_y, as_low_z, as_high_x,
     & as_high_y, as_high_z, as, at_low_x, at_low_y, at_low_z,
     & at_high_x, at_high_y, at_high_z, at, ab_low_x, ab_low_y,
     & ab_low_z, ab_high_x, ab_high_y, ab_high_z, ab, sew_low, sew_high
     & , sew, sns_low, sns_high, sns, stb_low, stb_high, stb, sewu_low,
     &  sewu_high, sewu, dxep_low, dxep_high, dxep, dxpw_low, dxpw_high
     & , dxpw, snsv_low, snsv_high, snsv, dynp_low, dynp_high, dynp,
     & dyps_low, dyps_high, dyps, stbw_low, stbw_high, stbw, dztp_low,
     & dztp_high, dztp, dzpb_low, dzpb_high, dzpb)

      implicit none
      integer idxLo(3)
      integer idxHi(3)
      integer den_low_x, den_low_y, den_low_z, den_high_x, den_high_y,
     & den_high_z
      double precision den(den_low_x:den_high_x, den_low_y:den_high_y,
     & den_low_z:den_high_z)
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
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer sewu_low
      integer sewu_high
      double precision sewu(sewu_low:sewu_high)
      integer dxep_low
      integer dxep_high
      double precision dxep(dxep_low:dxep_high)
      integer dxpw_low
      integer dxpw_high
      double precision dxpw(dxpw_low:dxpw_high)
      integer snsv_low
      integer snsv_high
      double precision snsv(snsv_low:snsv_high)
      integer dynp_low
      integer dynp_high
      double precision dynp(dynp_low:dynp_high)
      integer dyps_low
      integer dyps_high
      double precision dyps(dyps_low:dyps_high)
      integer stbw_low
      integer stbw_high
      double precision stbw(stbw_low:stbw_high)
      integer dztp_low
      integer dztp_high
      double precision dztp(dztp_low:dztp_high)
      integer dzpb_low
      integer dzpb_high
      double precision dzpb(dzpb_low:dzpb_high)
#endif /* __cplusplus */

#endif /* fspec_prescoef_var */
