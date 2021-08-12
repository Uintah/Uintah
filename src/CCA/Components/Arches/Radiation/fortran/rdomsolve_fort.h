
#ifndef fspec_rdomsolve
#define fspec_rdomsolve

#ifdef __cplusplus

#include <CCA/Components/Arches/Radiation/fortran/FortranNameMangle.h>

extern "C" void F_rdomsolve(int* idxlo,
                          int* idxhi,
                          int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                          int* ffield,
                          int* sew_low, int* sew_high, double* sew_ptr,
                          int* sns_low, int* sns_high, double* sns_ptr,
                          int* stb_low, int* stb_high, double* stb_ptr,
                          int* esrct_low_x, int* esrct_low_y, int* esrct_low_z, int* esrct_high_x, int* esrct_high_y, int* esrct_high_z, double* esrct_ptr,
                          int* l,
                          int* oxi_low, int* oxi_high, double* oxi_ptr,
                          int* omu_low, int* omu_high, double* omu_ptr,
                          int* oeta_low, int* oeta_high, double* oeta_ptr,
                          int* wt_low, int* wt_high, double* wt_ptr,
                          int* tg_low_x, int* tg_low_y, int* tg_low_z, int* tg_high_x, int* tg_high_y, int* tg_high_z, double* tg_ptr,
                          int* abskt_low_x, int* abskt_low_y, int* abskt_low_z, int* abskt_high_x, int* abskt_high_y, int* abskt_high_z, double* abskt_ptr,
                          int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                          int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
                          int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
                          int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
                          int* ap_low_x, int* ap_low_y, int* ap_low_z, int* ap_high_x, int* ap_high_y, int* ap_high_z, double* ap_ptr,
                          bool* plusX,
                          bool* plusY,
                          bool* plusZ,
                          int* fraction_low, int* fraction_high, double* fraction_ptr,
                          int* bands,
                          int* IncidentFluxE_low_x, int* IncidentFluxE_low_y, int* IncidentFluxE_low_z, int* IncidentFluxE_high_x, int* IncidentFluxE_high_y, int* IncidentFluxE_high_z, double* IncidentFluxE_ptr,
                          int* IncidentFluxW_low_x, int* IncidentFluxW_low_y, int* IncidentFluxW_low_z, int* IncidentFluxW_high_x, int* IncidentFluxW_high_y, int* IncidentFluxW_high_z, double* IncidentFluxW_ptr,
                          int* IncidentFluxN_low_x, int* IncidentFluxN_low_y, int* IncidentFluxN_low_z, int* IncidentFluxN_high_x, int* IncidentFluxN_high_y, int* IncidentFluxN_high_z, double* IncidentFluxN_ptr,
                          int* IncidentFluxS_low_x, int* IncidentFluxS_low_y, int* IncidentFluxS_low_z, int* IncidentFluxS_high_x, int* IncidentFluxS_high_y, int* IncidentFluxS_high_z, double* IncidentFluxS_ptr,
                          int* IncidentFluxT_low_x, int* IncidentFluxT_low_y, int* IncidentFluxT_low_z, int* IncidentFluxT_high_x, int* IncidentFluxT_high_y, int* IncidentFluxT_high_z, double* IncidentFluxT_ptr,
                          int* IncidentFluxB_low_x, int* IncidentFluxB_low_y, int* IncidentFluxB_low_z, int* IncidentFluxB_high_x, int* IncidentFluxB_high_y, int* IncidentFluxB_high_z, double* IncidentFluxB_ptr,
                          int* scatSrc_low_x, int* scatSrc_low_y, int* scatSrc_low_z, int* scatSrc_high_x, int* scatSrc_high_y, int* scatSrc_high_z, double* scatSrc_ptr);

static void fort_rdomsolve( Uintah::IntVector & idxlo,
                            Uintah::IntVector & idxhi,
                            Uintah::constCCVariable<int> & pcell,
                            int & ffield,
                            Uintah::OffsetArray1<double> & sew,
                            Uintah::OffsetArray1<double> & sns,
                            Uintah::OffsetArray1<double> & stb,
                            Uintah::CCVariable<double> & esrct,
                            int & l,
                            Uintah::OffsetArray1<double> & oxi,
                            Uintah::OffsetArray1<double> & omu,
                            Uintah::OffsetArray1<double> & oeta,
                            Uintah::OffsetArray1<double> & wt,
                            Uintah::constCCVariable<double> & tg,
                            Uintah::constCCVariable<double> & abskt,
                            Uintah::CCVariable<double> & su,
                            Uintah::CCVariable<double> & aw,
                            Uintah::CCVariable<double> & as,
                            Uintah::CCVariable<double> & ab,
                            Uintah::CCVariable<double> & ap,
                            bool & plusX,
                            bool & plusY,
                            bool & plusZ,
                            Uintah::OffsetArray1<double> & fraction,
                            int & bands,
                            Uintah::CCVariable<double> & IncidentFluxE,
                            Uintah::CCVariable<double> & IncidentFluxW,
                            Uintah::CCVariable<double> & IncidentFluxN,
                            Uintah::CCVariable<double> & IncidentFluxS,
                            Uintah::CCVariable<double> & IncidentFluxT,
                            Uintah::CCVariable<double> & IncidentFluxB,
                            Uintah::CCVariable<double> & scatSrc )
{
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  Uintah::IntVector esrct_low = esrct.getWindow()->getOffset();
  Uintah::IntVector esrct_high = esrct.getWindow()->getData()->size() + esrct_low - Uintah::IntVector(1, 1, 1);
  int esrct_low_x = esrct_low.x();
  int esrct_high_x = esrct_high.x();
  int esrct_low_y = esrct_low.y();
  int esrct_high_y = esrct_high.y();
  int esrct_low_z = esrct_low.z();
  int esrct_high_z = esrct_high.z();
  int oxi_low = oxi.low();
  int oxi_high = oxi.high();
  int omu_low = omu.low();
  int omu_high = omu.high();
  int oeta_low = oeta.low();
  int oeta_high = oeta.high();
  int wt_low = wt.low();
  int wt_high = wt.high();
  Uintah::IntVector tg_low = tg.getWindow()->getOffset();
  Uintah::IntVector tg_high = tg.getWindow()->getData()->size() + tg_low - Uintah::IntVector(1, 1, 1);
  int tg_low_x = tg_low.x();
  int tg_high_x = tg_high.x();
  int tg_low_y = tg_low.y();
  int tg_high_y = tg_high.y();
  int tg_low_z = tg_low.z();
  int tg_high_z = tg_high.z();
  Uintah::IntVector abskt_low = abskt.getWindow()->getOffset();
  Uintah::IntVector abskt_high = abskt.getWindow()->getData()->size() + abskt_low - Uintah::IntVector(1, 1, 1);
  int abskt_low_x = abskt_low.x();
  int abskt_high_x = abskt_high.x();
  int abskt_low_y = abskt_low.y();
  int abskt_high_y = abskt_high.y();
  int abskt_low_z = abskt_low.z();
  int abskt_high_z = abskt_high.z();
  Uintah::IntVector su_low = su.getWindow()->getOffset();
  Uintah::IntVector su_high = su.getWindow()->getData()->size() + su_low - Uintah::IntVector(1, 1, 1);
  int su_low_x = su_low.x();
  int su_high_x = su_high.x();
  int su_low_y = su_low.y();
  int su_high_y = su_high.y();
  int su_low_z = su_low.z();
  int su_high_z = su_high.z();
  Uintah::IntVector aw_low = aw.getWindow()->getOffset();
  Uintah::IntVector aw_high = aw.getWindow()->getData()->size() + aw_low - Uintah::IntVector(1, 1, 1);
  int aw_low_x = aw_low.x();
  int aw_high_x = aw_high.x();
  int aw_low_y = aw_low.y();
  int aw_high_y = aw_high.y();
  int aw_low_z = aw_low.z();
  int aw_high_z = aw_high.z();
  Uintah::IntVector as_low = as.getWindow()->getOffset();
  Uintah::IntVector as_high = as.getWindow()->getData()->size() + as_low - Uintah::IntVector(1, 1, 1);
  int as_low_x = as_low.x();
  int as_high_x = as_high.x();
  int as_low_y = as_low.y();
  int as_high_y = as_high.y();
  int as_low_z = as_low.z();
  int as_high_z = as_high.z();
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
  int fraction_low = fraction.low();
  int fraction_high = fraction.high();
  Uintah::IntVector IncidentFluxE_low = IncidentFluxE.getWindow()->getOffset();
  Uintah::IntVector IncidentFluxE_high = IncidentFluxE.getWindow()->getData()->size() + IncidentFluxE_low - Uintah::IntVector(1, 1, 1);
  int IncidentFluxE_low_x = IncidentFluxE_low.x();
  int IncidentFluxE_high_x = IncidentFluxE_high.x();
  int IncidentFluxE_low_y = IncidentFluxE_low.y();
  int IncidentFluxE_high_y = IncidentFluxE_high.y();
  int IncidentFluxE_low_z = IncidentFluxE_low.z();
  int IncidentFluxE_high_z = IncidentFluxE_high.z();
  Uintah::IntVector IncidentFluxW_low = IncidentFluxW.getWindow()->getOffset();
  Uintah::IntVector IncidentFluxW_high = IncidentFluxW.getWindow()->getData()->size() + IncidentFluxW_low - Uintah::IntVector(1, 1, 1);
  int IncidentFluxW_low_x = IncidentFluxW_low.x();
  int IncidentFluxW_high_x = IncidentFluxW_high.x();
  int IncidentFluxW_low_y = IncidentFluxW_low.y();
  int IncidentFluxW_high_y = IncidentFluxW_high.y();
  int IncidentFluxW_low_z = IncidentFluxW_low.z();
  int IncidentFluxW_high_z = IncidentFluxW_high.z();
  Uintah::IntVector IncidentFluxN_low = IncidentFluxN.getWindow()->getOffset();
  Uintah::IntVector IncidentFluxN_high = IncidentFluxN.getWindow()->getData()->size() + IncidentFluxN_low - Uintah::IntVector(1, 1, 1);
  int IncidentFluxN_low_x = IncidentFluxN_low.x();
  int IncidentFluxN_high_x = IncidentFluxN_high.x();
  int IncidentFluxN_low_y = IncidentFluxN_low.y();
  int IncidentFluxN_high_y = IncidentFluxN_high.y();
  int IncidentFluxN_low_z = IncidentFluxN_low.z();
  int IncidentFluxN_high_z = IncidentFluxN_high.z();
  Uintah::IntVector IncidentFluxS_low = IncidentFluxS.getWindow()->getOffset();
  Uintah::IntVector IncidentFluxS_high = IncidentFluxS.getWindow()->getData()->size() + IncidentFluxS_low - Uintah::IntVector(1, 1, 1);
  int IncidentFluxS_low_x = IncidentFluxS_low.x();
  int IncidentFluxS_high_x = IncidentFluxS_high.x();
  int IncidentFluxS_low_y = IncidentFluxS_low.y();
  int IncidentFluxS_high_y = IncidentFluxS_high.y();
  int IncidentFluxS_low_z = IncidentFluxS_low.z();
  int IncidentFluxS_high_z = IncidentFluxS_high.z();
  Uintah::IntVector IncidentFluxT_low = IncidentFluxT.getWindow()->getOffset();
  Uintah::IntVector IncidentFluxT_high = IncidentFluxT.getWindow()->getData()->size() + IncidentFluxT_low - Uintah::IntVector(1, 1, 1);
  int IncidentFluxT_low_x = IncidentFluxT_low.x();
  int IncidentFluxT_high_x = IncidentFluxT_high.x();
  int IncidentFluxT_low_y = IncidentFluxT_low.y();
  int IncidentFluxT_high_y = IncidentFluxT_high.y();
  int IncidentFluxT_low_z = IncidentFluxT_low.z();
  int IncidentFluxT_high_z = IncidentFluxT_high.z();
  Uintah::IntVector IncidentFluxB_low = IncidentFluxB.getWindow()->getOffset();
  Uintah::IntVector IncidentFluxB_high = IncidentFluxB.getWindow()->getData()->size() + IncidentFluxB_low - Uintah::IntVector(1, 1, 1);
  int IncidentFluxB_low_x = IncidentFluxB_low.x();
  int IncidentFluxB_high_x = IncidentFluxB_high.x();
  int IncidentFluxB_low_y = IncidentFluxB_low.y();
  int IncidentFluxB_high_y = IncidentFluxB_high.y();
  int IncidentFluxB_low_z = IncidentFluxB_low.z();
  int IncidentFluxB_high_z = IncidentFluxB_high.z();
  Uintah::IntVector scatSrc_low = scatSrc.getWindow()->getOffset();
  Uintah::IntVector scatSrc_high = scatSrc.getWindow()->getData()->size() + scatSrc_low - Uintah::IntVector(1, 1, 1);
  int scatSrc_low_x = scatSrc_low.x();
  int scatSrc_high_x = scatSrc_high.x();
  int scatSrc_low_y = scatSrc_low.y();
  int scatSrc_high_y = scatSrc_high.y();
  int scatSrc_low_z = scatSrc_low.z();
  int scatSrc_high_z = scatSrc_high.z();
  F_rdomsolve( idxlo.get_pointer(),
             idxhi.get_pointer(),
             &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
             &ffield,
             &sew_low, &sew_high, sew.get_objs(),
             &sns_low, &sns_high, sns.get_objs(),
             &stb_low, &stb_high, stb.get_objs(),
             &esrct_low_x, &esrct_low_y, &esrct_low_z, &esrct_high_x, &esrct_high_y, &esrct_high_z, esrct.getPointer(),
             &l,
             &oxi_low, &oxi_high, oxi.get_objs(),
             &omu_low, &omu_high, omu.get_objs(),
             &oeta_low, &oeta_high, oeta.get_objs(),
             &wt_low, &wt_high, wt.get_objs(),
             &tg_low_x, &tg_low_y, &tg_low_z, &tg_high_x, &tg_high_y, &tg_high_z, const_cast<double*>(tg.getPointer()),
             &abskt_low_x, &abskt_low_y, &abskt_low_z, &abskt_high_x, &abskt_high_y, &abskt_high_z, const_cast<double*>(abskt.getPointer()),
             &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
             &aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
             &as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
             &ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
             &ap_low_x, &ap_low_y, &ap_low_z, &ap_high_x, &ap_high_y, &ap_high_z, ap.getPointer(),
             &plusX,
             &plusY,
             &plusZ,
             &fraction_low, &fraction_high, fraction.get_objs(),
             &bands,
             &IncidentFluxE_low_x, &IncidentFluxE_low_y, &IncidentFluxE_low_z, &IncidentFluxE_high_x, &IncidentFluxE_high_y, &IncidentFluxE_high_z, IncidentFluxE.getPointer(),
             &IncidentFluxW_low_x, &IncidentFluxW_low_y, &IncidentFluxW_low_z, &IncidentFluxW_high_x, &IncidentFluxW_high_y, &IncidentFluxW_high_z, IncidentFluxW.getPointer(),
             &IncidentFluxN_low_x, &IncidentFluxN_low_y, &IncidentFluxN_low_z, &IncidentFluxN_high_x, &IncidentFluxN_high_y, &IncidentFluxN_high_z, IncidentFluxN.getPointer(),
             &IncidentFluxS_low_x, &IncidentFluxS_low_y, &IncidentFluxS_low_z, &IncidentFluxS_high_x, &IncidentFluxS_high_y, &IncidentFluxS_high_z, IncidentFluxS.getPointer(),
             &IncidentFluxT_low_x, &IncidentFluxT_low_y, &IncidentFluxT_low_z, &IncidentFluxT_high_x, &IncidentFluxT_high_y, &IncidentFluxT_high_z, IncidentFluxT.getPointer(),
             &IncidentFluxB_low_x, &IncidentFluxB_low_y, &IncidentFluxB_low_z, &IncidentFluxB_high_x, &IncidentFluxB_high_y, &IncidentFluxB_high_z, IncidentFluxB.getPointer(),
             &scatSrc_low_x, &scatSrc_low_y, &scatSrc_low_z, &scatSrc_high_x, &scatSrc_high_y, &scatSrc_high_z, scatSrc.getPointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine RDOMSOLVE(idxlo, idxhi, pcell_low_x, pcell_low_y,
     & pcell_low_z, pcell_high_x, pcell_high_y, pcell_high_z, pcell, 
     & ffield, sew_low, sew_high, sew, sns_low, sns_high, sns, stb_low,
     &  stb_high, stb, esrct_low_x, esrct_low_y, esrct_low_z, 
     & esrct_high_x, esrct_high_y, esrct_high_z, esrct, l, oxi_low, 
     & oxi_high, oxi, omu_low, omu_high, omu, oeta_low, oeta_high, oeta
     & , wt_low, wt_high, wt, tg_low_x, tg_low_y, tg_low_z, tg_high_x, 
     & tg_high_y, tg_high_z, tg, abskt_low_x, abskt_low_y, abskt_low_z,
     &  abskt_high_x, abskt_high_y, abskt_high_z, abskt, su_low_x, 
     & su_low_y, su_low_z, su_high_x, su_high_y, su_high_z, su, 
     & aw_low_x, aw_low_y, aw_low_z, aw_high_x, aw_high_y, aw_high_z, 
     & aw, as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, 
     & as_high_z, as, ab_low_x, ab_low_y, ab_low_z, ab_high_x, 
     & ab_high_y, ab_high_z, ab, ap_low_x, ap_low_y, ap_low_z, 
     & ap_high_x, ap_high_y, ap_high_z, ap, plusX, plusY, plusZ, 
     & fraction_low, fraction_high, fraction, bands, 
     & IncidentFluxE_low_x, IncidentFluxE_low_y, IncidentFluxE_low_z, 
     & IncidentFluxE_high_x, IncidentFluxE_high_y, IncidentFluxE_high_z
     & , IncidentFluxE, IncidentFluxW_low_x, IncidentFluxW_low_y, 
     & IncidentFluxW_low_z, IncidentFluxW_high_x, IncidentFluxW_high_y,
     &  IncidentFluxW_high_z, IncidentFluxW, IncidentFluxN_low_x, 
     & IncidentFluxN_low_y, IncidentFluxN_low_z, IncidentFluxN_high_x, 
     & IncidentFluxN_high_y, IncidentFluxN_high_z, IncidentFluxN, 
     & IncidentFluxS_low_x, IncidentFluxS_low_y, IncidentFluxS_low_z, 
     & IncidentFluxS_high_x, IncidentFluxS_high_y, IncidentFluxS_high_z
     & , IncidentFluxS, IncidentFluxT_low_x, IncidentFluxT_low_y, 
     & IncidentFluxT_low_z, IncidentFluxT_high_x, IncidentFluxT_high_y,
     &  IncidentFluxT_high_z, IncidentFluxT, IncidentFluxB_low_x, 
     & IncidentFluxB_low_y, IncidentFluxB_low_z, IncidentFluxB_high_x, 
     & IncidentFluxB_high_y, IncidentFluxB_high_z, IncidentFluxB, 
     & scatSrc_low_x, scatSrc_low_y, scatSrc_low_z, scatSrc_high_x, 
     & scatSrc_high_y, scatSrc_high_z, scatSrc)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer ffield
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer esrct_low_x, esrct_low_y, esrct_low_z, esrct_high_x, 
     & esrct_high_y, esrct_high_z
      double precision esrct(esrct_low_x:esrct_high_x, esrct_low_y:
     & esrct_high_y, esrct_low_z:esrct_high_z)
      integer l
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
      integer tg_low_x, tg_low_y, tg_low_z, tg_high_x, tg_high_y, 
     & tg_high_z
      double precision tg(tg_low_x:tg_high_x, tg_low_y:tg_high_y, 
     & tg_low_z:tg_high_z)
      integer abskt_low_x, abskt_low_y, abskt_low_z, abskt_high_x, 
     & abskt_high_y, abskt_high_z
      double precision abskt(abskt_low_x:abskt_high_x, abskt_low_y:
     & abskt_high_y, abskt_low_z:abskt_high_z)
      integer su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z
      double precision su(su_low_x:su_high_x, su_low_y:su_high_y, 
     & su_low_z:su_high_z)
      integer aw_low_x, aw_low_y, aw_low_z, aw_high_x, aw_high_y, 
     & aw_high_z
      double precision aw(aw_low_x:aw_high_x, aw_low_y:aw_high_y, 
     & aw_low_z:aw_high_z)
      integer as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, 
     & as_high_z
      double precision as(as_low_x:as_high_x, as_low_y:as_high_y, 
     & as_low_z:as_high_z)
      integer ab_low_x, ab_low_y, ab_low_z, ab_high_x, ab_high_y, 
     & ab_high_z
      double precision ab(ab_low_x:ab_high_x, ab_low_y:ab_high_y, 
     & ab_low_z:ab_high_z)
      integer ap_low_x, ap_low_y, ap_low_z, ap_high_x, ap_high_y, 
     & ap_high_z
      double precision ap(ap_low_x:ap_high_x, ap_low_y:ap_high_y, 
     & ap_low_z:ap_high_z)
      logical*1 plusX
      logical*1 plusY
      logical*1 plusZ
      integer fraction_low
      integer fraction_high
      double precision fraction(fraction_low:fraction_high)
      integer bands
      integer IncidentFluxE_low_x, IncidentFluxE_low_y, 
     & IncidentFluxE_low_z, IncidentFluxE_high_x, IncidentFluxE_high_y,
     &  IncidentFluxE_high_z
      double precision IncidentFluxE(IncidentFluxE_low_x:
     & IncidentFluxE_high_x, IncidentFluxE_low_y:IncidentFluxE_high_y, 
     & IncidentFluxE_low_z:IncidentFluxE_high_z)
      integer IncidentFluxW_low_x, IncidentFluxW_low_y, 
     & IncidentFluxW_low_z, IncidentFluxW_high_x, IncidentFluxW_high_y,
     &  IncidentFluxW_high_z
      double precision IncidentFluxW(IncidentFluxW_low_x:
     & IncidentFluxW_high_x, IncidentFluxW_low_y:IncidentFluxW_high_y, 
     & IncidentFluxW_low_z:IncidentFluxW_high_z)
      integer IncidentFluxN_low_x, IncidentFluxN_low_y, 
     & IncidentFluxN_low_z, IncidentFluxN_high_x, IncidentFluxN_high_y,
     &  IncidentFluxN_high_z
      double precision IncidentFluxN(IncidentFluxN_low_x:
     & IncidentFluxN_high_x, IncidentFluxN_low_y:IncidentFluxN_high_y, 
     & IncidentFluxN_low_z:IncidentFluxN_high_z)
      integer IncidentFluxS_low_x, IncidentFluxS_low_y, 
     & IncidentFluxS_low_z, IncidentFluxS_high_x, IncidentFluxS_high_y,
     &  IncidentFluxS_high_z
      double precision IncidentFluxS(IncidentFluxS_low_x:
     & IncidentFluxS_high_x, IncidentFluxS_low_y:IncidentFluxS_high_y, 
     & IncidentFluxS_low_z:IncidentFluxS_high_z)
      integer IncidentFluxT_low_x, IncidentFluxT_low_y, 
     & IncidentFluxT_low_z, IncidentFluxT_high_x, IncidentFluxT_high_y,
     &  IncidentFluxT_high_z
      double precision IncidentFluxT(IncidentFluxT_low_x:
     & IncidentFluxT_high_x, IncidentFluxT_low_y:IncidentFluxT_high_y, 
     & IncidentFluxT_low_z:IncidentFluxT_high_z)
      integer IncidentFluxB_low_x, IncidentFluxB_low_y, 
     & IncidentFluxB_low_z, IncidentFluxB_high_x, IncidentFluxB_high_y,
     &  IncidentFluxB_high_z
      double precision IncidentFluxB(IncidentFluxB_low_x:
     & IncidentFluxB_high_x, IncidentFluxB_low_y:IncidentFluxB_high_y, 
     & IncidentFluxB_low_z:IncidentFluxB_high_z)
      integer scatSrc_low_x, scatSrc_low_y, scatSrc_low_z, 
     & scatSrc_high_x, scatSrc_high_y, scatSrc_high_z
      double precision scatSrc(scatSrc_low_x:scatSrc_high_x, 
     & scatSrc_low_y:scatSrc_high_y, scatSrc_low_z:scatSrc_high_z)
#endif /* __cplusplus */

#endif /* fspec_rdomsolve */
