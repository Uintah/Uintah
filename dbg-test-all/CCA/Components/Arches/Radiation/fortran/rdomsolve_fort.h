
#ifndef fspec_rdomsolve
#define fspec_rdomsolve

#ifdef __cplusplus

extern "C" void rdomsolve_(int* idxlo,
                           int* idxhi,
                           int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                           int* ffield,
                           int* sew_low, int* sew_high, double* sew_ptr,
                           int* sns_low, int* sns_high, double* sns_ptr,
                           int* stb_low, int* stb_high, double* stb_ptr,
                           int* esrcg_low_x, int* esrcg_low_y, int* esrcg_low_z, int* esrcg_high_x, int* esrcg_high_y, int* esrcg_high_z, double* esrcg_ptr,
                           int* l,
                           int* oxi_low, int* oxi_high, double* oxi_ptr,
                           int* omu_low, int* omu_high, double* omu_ptr,
                           int* oeta_low, int* oeta_high, double* oeta_ptr,
                           int* wt_low, int* wt_high, double* wt_ptr,
                           int* tg_low_x, int* tg_low_y, int* tg_low_z, int* tg_high_x, int* tg_high_y, int* tg_high_z, double* tg_ptr,
                           int* abskg_low_x, int* abskg_low_y, int* abskg_low_z, int* abskg_high_x, int* abskg_high_y, int* abskg_high_z, double* abskg_ptr,
                           int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
                           int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
                           int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
                           int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
                           int* ap_low_x, int* ap_low_y, int* ap_low_z, int* ap_high_x, int* ap_high_y, int* ap_high_z, double* ap_ptr,
                           int* ae_low_x, int* ae_low_y, int* ae_low_z, int* ae_high_x, int* ae_high_y, int* ae_high_z, double* ae_ptr,
                           int* an_low_x, int* an_low_y, int* an_low_z, int* an_high_x, int* an_high_y, int* an_high_z, double* an_ptr,
                           int* at_low_x, int* at_low_y, int* at_low_z, int* at_high_x, int* at_high_y, int* at_high_z, double* at_ptr,
                           bool* plusX,
                           bool* plusY,
                           bool* plusZ,
                           int* fraction_low, int* fraction_high, double* fraction_ptr,
                           int* bands,
                           double* intrusion_abskg);

static void fort_rdomsolve( Uintah::IntVector & idxlo,
                            Uintah::IntVector & idxhi,
                            Uintah::constCCVariable<int> & pcell,
                            int & ffield,
                            Uintah::OffsetArray1<double> & sew,
                            Uintah::OffsetArray1<double> & sns,
                            Uintah::OffsetArray1<double> & stb,
                            Uintah::CCVariable<double> & esrcg,
                            int & l,
                            Uintah::OffsetArray1<double> & oxi,
                            Uintah::OffsetArray1<double> & omu,
                            Uintah::OffsetArray1<double> & oeta,
                            Uintah::OffsetArray1<double> & wt,
                            Uintah::constCCVariable<double> & tg,
                            Uintah::constCCVariable<double> & abskg,
                            Uintah::CCVariable<double> & su,
                            Uintah::CCVariable<double> & aw,
                            Uintah::CCVariable<double> & as,
                            Uintah::CCVariable<double> & ab,
                            Uintah::CCVariable<double> & ap,
                            Uintah::CCVariable<double> & ae,
                            Uintah::CCVariable<double> & an,
                            Uintah::CCVariable<double> & at,
                            bool & plusX,
                            bool & plusY,
                            bool & plusZ,
                            Uintah::OffsetArray1<double> & fraction,
                            int & bands,
                            double & intrusion_abskg )
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
  Uintah::IntVector esrcg_low = esrcg.getWindow()->getOffset();
  Uintah::IntVector esrcg_high = esrcg.getWindow()->getData()->size() + esrcg_low - Uintah::IntVector(1, 1, 1);
  int esrcg_low_x = esrcg_low.x();
  int esrcg_high_x = esrcg_high.x();
  int esrcg_low_y = esrcg_low.y();
  int esrcg_high_y = esrcg_high.y();
  int esrcg_low_z = esrcg_low.z();
  int esrcg_high_z = esrcg_high.z();
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
  Uintah::IntVector abskg_low = abskg.getWindow()->getOffset();
  Uintah::IntVector abskg_high = abskg.getWindow()->getData()->size() + abskg_low - Uintah::IntVector(1, 1, 1);
  int abskg_low_x = abskg_low.x();
  int abskg_high_x = abskg_high.x();
  int abskg_low_y = abskg_low.y();
  int abskg_high_y = abskg_high.y();
  int abskg_low_z = abskg_low.z();
  int abskg_high_z = abskg_high.z();
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
  Uintah::IntVector ae_low = ae.getWindow()->getOffset();
  Uintah::IntVector ae_high = ae.getWindow()->getData()->size() + ae_low - Uintah::IntVector(1, 1, 1);
  int ae_low_x = ae_low.x();
  int ae_high_x = ae_high.x();
  int ae_low_y = ae_low.y();
  int ae_high_y = ae_high.y();
  int ae_low_z = ae_low.z();
  int ae_high_z = ae_high.z();
  Uintah::IntVector an_low = an.getWindow()->getOffset();
  Uintah::IntVector an_high = an.getWindow()->getData()->size() + an_low - Uintah::IntVector(1, 1, 1);
  int an_low_x = an_low.x();
  int an_high_x = an_high.x();
  int an_low_y = an_low.y();
  int an_high_y = an_high.y();
  int an_low_z = an_low.z();
  int an_high_z = an_high.z();
  Uintah::IntVector at_low = at.getWindow()->getOffset();
  Uintah::IntVector at_high = at.getWindow()->getData()->size() + at_low - Uintah::IntVector(1, 1, 1);
  int at_low_x = at_low.x();
  int at_high_x = at_high.x();
  int at_low_y = at_low.y();
  int at_high_y = at_high.y();
  int at_low_z = at_low.z();
  int at_high_z = at_high.z();
  int fraction_low = fraction.low();
  int fraction_high = fraction.high();
  rdomsolve_( idxlo.get_pointer(),
              idxhi.get_pointer(),
              &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
              &ffield,
              &sew_low, &sew_high, sew.get_objs(),
              &sns_low, &sns_high, sns.get_objs(),
              &stb_low, &stb_high, stb.get_objs(),
              &esrcg_low_x, &esrcg_low_y, &esrcg_low_z, &esrcg_high_x, &esrcg_high_y, &esrcg_high_z, esrcg.getPointer(),
              &l,
              &oxi_low, &oxi_high, oxi.get_objs(),
              &omu_low, &omu_high, omu.get_objs(),
              &oeta_low, &oeta_high, oeta.get_objs(),
              &wt_low, &wt_high, wt.get_objs(),
              &tg_low_x, &tg_low_y, &tg_low_z, &tg_high_x, &tg_high_y, &tg_high_z, const_cast<double*>(tg.getPointer()),
              &abskg_low_x, &abskg_low_y, &abskg_low_z, &abskg_high_x, &abskg_high_y, &abskg_high_z, const_cast<double*>(abskg.getPointer()),
              &su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
              &aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
              &as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
              &ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
              &ap_low_x, &ap_low_y, &ap_low_z, &ap_high_x, &ap_high_y, &ap_high_z, ap.getPointer(),
              &ae_low_x, &ae_low_y, &ae_low_z, &ae_high_x, &ae_high_y, &ae_high_z, ae.getPointer(),
              &an_low_x, &an_low_y, &an_low_z, &an_high_x, &an_high_y, &an_high_z, an.getPointer(),
              &at_low_x, &at_low_y, &at_low_z, &at_high_x, &at_high_y, &at_high_z, at.getPointer(),
              &plusX,
              &plusY,
              &plusZ,
              &fraction_low, &fraction_high, fraction.get_objs(),
              &bands,
              &intrusion_abskg );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine rdomsolve(idxlo, idxhi, pcell_low_x, pcell_low_y, 
     & pcell_low_z, pcell_high_x, pcell_high_y, pcell_high_z, pcell, 
     & ffield, sew_low, sew_high, sew, sns_low, sns_high, sns, stb_low,
     &  stb_high, stb, esrcg_low_x, esrcg_low_y, esrcg_low_z, 
     & esrcg_high_x, esrcg_high_y, esrcg_high_z, esrcg, l, oxi_low, 
     & oxi_high, oxi, omu_low, omu_high, omu, oeta_low, oeta_high, oeta
     & , wt_low, wt_high, wt, tg_low_x, tg_low_y, tg_low_z, tg_high_x, 
     & tg_high_y, tg_high_z, tg, abskg_low_x, abskg_low_y, abskg_low_z,
     &  abskg_high_x, abskg_high_y, abskg_high_z, abskg, su_low_x, 
     & su_low_y, su_low_z, su_high_x, su_high_y, su_high_z, su, 
     & aw_low_x, aw_low_y, aw_low_z, aw_high_x, aw_high_y, aw_high_z, 
     & aw, as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, 
     & as_high_z, as, ab_low_x, ab_low_y, ab_low_z, ab_high_x, 
     & ab_high_y, ab_high_z, ab, ap_low_x, ap_low_y, ap_low_z, 
     & ap_high_x, ap_high_y, ap_high_z, ap, ae_low_x, ae_low_y, 
     & ae_low_z, ae_high_x, ae_high_y, ae_high_z, ae, an_low_x, 
     & an_low_y, an_low_z, an_high_x, an_high_y, an_high_z, an, 
     & at_low_x, at_low_y, at_low_z, at_high_x, at_high_y, at_high_z, 
     & at, plusX, plusY, plusZ, fraction_low, fraction_high, fraction, 
     & bands, intrusion_abskg)

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
      integer esrcg_low_x, esrcg_low_y, esrcg_low_z, esrcg_high_x, 
     & esrcg_high_y, esrcg_high_z
      double precision esrcg(esrcg_low_x:esrcg_high_x, esrcg_low_y:
     & esrcg_high_y, esrcg_low_z:esrcg_high_z)
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
      integer abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z
      double precision abskg(abskg_low_x:abskg_high_x, abskg_low_y:
     & abskg_high_y, abskg_low_z:abskg_high_z)
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
      integer ae_low_x, ae_low_y, ae_low_z, ae_high_x, ae_high_y, 
     & ae_high_z
      double precision ae(ae_low_x:ae_high_x, ae_low_y:ae_high_y, 
     & ae_low_z:ae_high_z)
      integer an_low_x, an_low_y, an_low_z, an_high_x, an_high_y, 
     & an_high_z
      double precision an(an_low_x:an_high_x, an_low_y:an_high_y, 
     & an_low_z:an_high_z)
      integer at_low_x, at_low_y, at_low_z, at_high_x, at_high_y, 
     & at_high_z
      double precision at(at_low_x:at_high_x, at_low_y:at_high_y, 
     & at_low_z:at_high_z)
      logical*1 plusX
      logical*1 plusY
      logical*1 plusZ
      integer fraction_low
      integer fraction_high
      double precision fraction(fraction_low:fraction_high)
      integer bands
      double precision intrusion_abskg
#endif /* __cplusplus */

#endif /* fspec_rdomsolve */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
