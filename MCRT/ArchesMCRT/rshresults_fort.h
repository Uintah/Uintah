
#ifndef fspec_rshresults
#define fspec_rshresults

#ifdef __cplusplus

extern "C" void rshresults_(int* idxlo,
int* idxhi,
int* cenint_low_x, int* cenint_low_y, int* cenint_low_z, int* cenint_high_x, int* cenint_high_y, int* cenint_high_z, double* cenint_ptr,
int* volq_low_x, int* volq_low_y, int* volq_low_z, int* volq_high_x, int* volq_high_y, int* volq_high_z, double* volq_ptr,
int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
int* ffield,
int* xx_low, int* xx_high, double* xx_ptr,
int* yy_low, int* yy_high, double* yy_ptr,
int* zz_low, int* zz_high, double* zz_ptr,
int* tg_low_x, int* tg_low_y, int* tg_low_z, int* tg_high_x, int* tg_high_y, int* tg_high_z, double* tg_ptr,
int* qfluxe_low_x, int* qfluxe_low_y, int* qfluxe_low_z, int* qfluxe_high_x, int* qfluxe_high_y, int* qfluxe_high_z, double* qfluxe_ptr,
int* qfluxw_low_x, int* qfluxw_low_y, int* qfluxw_low_z, int* qfluxw_high_x, int* qfluxw_high_y, int* qfluxw_high_z, double* qfluxw_ptr,
int* qfluxn_low_x, int* qfluxn_low_y, int* qfluxn_low_z, int* qfluxn_high_x, int* qfluxn_high_y, int* qfluxn_high_z, double* qfluxn_ptr,
int* qfluxs_low_x, int* qfluxs_low_y, int* qfluxs_low_z, int* qfluxs_high_x, int* qfluxs_high_y, int* qfluxs_high_z, double* qfluxs_ptr,
int* qfluxt_low_x, int* qfluxt_low_y, int* qfluxt_low_z, int* qfluxt_high_x, int* qfluxt_high_y, int* qfluxt_high_z, double* qfluxt_ptr,
int* qfluxb_low_x, int* qfluxb_low_y, int* qfluxb_low_z, int* qfluxb_high_x, int* qfluxb_high_y, int* qfluxb_high_z, double* qfluxb_ptr,
int* abskg_low_x, int* abskg_low_y, int* abskg_low_z, int* abskg_high_x, int* abskg_high_y, int* abskg_high_z, double* abskg_ptr,
int* shgamma_low_x, int* shgamma_low_y, int* shgamma_low_z, int* shgamma_high_x, int* shgamma_high_y, int* shgamma_high_z, double* shgamma_ptr,
int* esrcg_low_x, int* esrcg_low_y, int* esrcg_low_z, int* esrcg_high_x, int* esrcg_high_y, int* esrcg_high_z, double* esrcg_ptr,
int* src_low_x, int* src_low_y, int* src_low_z, int* src_high_x, int* src_high_y, int* src_high_z, double* src_ptr,
int* fraction_low, int* fraction_high, double* fraction_ptr,
int* fractiontwo_low, int* fractiontwo_high, double* fractiontwo_ptr,
int* bands);

static void fort_rshresults(IntVector& idxlo,
IntVector& idxhi,
CCVariable<double>& cenint,
CCVariable<double>& volq,
constCCVariable<int>& pcell,
int& ffield,
OffsetArray1<double>& xx,
OffsetArray1<double>& yy,
OffsetArray1<double>& zz,
CCVariable<double>& tg,
CCVariable<double>& qfluxe,
CCVariable<double>& qfluxw,
CCVariable<double>& qfluxn,
CCVariable<double>& qfluxs,
CCVariable<double>& qfluxt,
CCVariable<double>& qfluxb,
CCVariable<double>& abskg,
CCVariable<double>& shgamma,
CCVariable<double>& esrcg,
CCVariable<double>& src,
OffsetArray1<double>& fraction,
OffsetArray1<double>& fractiontwo,
int& bands)
{
  IntVector cenint_low = cenint.getWindow()->getOffset();
  IntVector cenint_high = cenint.getWindow()->getData()->size() + cenint_low - IntVector(1, 1, 1);
  int cenint_low_x = cenint_low.x();
  int cenint_high_x = cenint_high.x();
  int cenint_low_y = cenint_low.y();
  int cenint_high_y = cenint_high.y();
  int cenint_low_z = cenint_low.z();
  int cenint_high_z = cenint_high.z();
  IntVector volq_low = volq.getWindow()->getOffset();
  IntVector volq_high = volq.getWindow()->getData()->size() + volq_low - IntVector(1, 1, 1);
  int volq_low_x = volq_low.x();
  int volq_high_x = volq_high.x();
  int volq_low_y = volq_low.y();
  int volq_high_y = volq_high.y();
  int volq_low_z = volq_low.z();
  int volq_high_z = volq_high.z();
  IntVector pcell_low = pcell.getWindow()->getOffset();
  IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  int xx_low = xx.low();
  int xx_high = xx.high();
  int yy_low = yy.low();
  int yy_high = yy.high();
  int zz_low = zz.low();
  int zz_high = zz.high();
  IntVector tg_low = tg.getWindow()->getOffset();
  IntVector tg_high = tg.getWindow()->getData()->size() + tg_low - IntVector(1, 1, 1);
  int tg_low_x = tg_low.x();
  int tg_high_x = tg_high.x();
  int tg_low_y = tg_low.y();
  int tg_high_y = tg_high.y();
  int tg_low_z = tg_low.z();
  int tg_high_z = tg_high.z();
  IntVector qfluxe_low = qfluxe.getWindow()->getOffset();
  IntVector qfluxe_high = qfluxe.getWindow()->getData()->size() + qfluxe_low - IntVector(1, 1, 1);
  int qfluxe_low_x = qfluxe_low.x();
  int qfluxe_high_x = qfluxe_high.x();
  int qfluxe_low_y = qfluxe_low.y();
  int qfluxe_high_y = qfluxe_high.y();
  int qfluxe_low_z = qfluxe_low.z();
  int qfluxe_high_z = qfluxe_high.z();
  IntVector qfluxw_low = qfluxw.getWindow()->getOffset();
  IntVector qfluxw_high = qfluxw.getWindow()->getData()->size() + qfluxw_low - IntVector(1, 1, 1);
  int qfluxw_low_x = qfluxw_low.x();
  int qfluxw_high_x = qfluxw_high.x();
  int qfluxw_low_y = qfluxw_low.y();
  int qfluxw_high_y = qfluxw_high.y();
  int qfluxw_low_z = qfluxw_low.z();
  int qfluxw_high_z = qfluxw_high.z();
  IntVector qfluxn_low = qfluxn.getWindow()->getOffset();
  IntVector qfluxn_high = qfluxn.getWindow()->getData()->size() + qfluxn_low - IntVector(1, 1, 1);
  int qfluxn_low_x = qfluxn_low.x();
  int qfluxn_high_x = qfluxn_high.x();
  int qfluxn_low_y = qfluxn_low.y();
  int qfluxn_high_y = qfluxn_high.y();
  int qfluxn_low_z = qfluxn_low.z();
  int qfluxn_high_z = qfluxn_high.z();
  IntVector qfluxs_low = qfluxs.getWindow()->getOffset();
  IntVector qfluxs_high = qfluxs.getWindow()->getData()->size() + qfluxs_low - IntVector(1, 1, 1);
  int qfluxs_low_x = qfluxs_low.x();
  int qfluxs_high_x = qfluxs_high.x();
  int qfluxs_low_y = qfluxs_low.y();
  int qfluxs_high_y = qfluxs_high.y();
  int qfluxs_low_z = qfluxs_low.z();
  int qfluxs_high_z = qfluxs_high.z();
  IntVector qfluxt_low = qfluxt.getWindow()->getOffset();
  IntVector qfluxt_high = qfluxt.getWindow()->getData()->size() + qfluxt_low - IntVector(1, 1, 1);
  int qfluxt_low_x = qfluxt_low.x();
  int qfluxt_high_x = qfluxt_high.x();
  int qfluxt_low_y = qfluxt_low.y();
  int qfluxt_high_y = qfluxt_high.y();
  int qfluxt_low_z = qfluxt_low.z();
  int qfluxt_high_z = qfluxt_high.z();
  IntVector qfluxb_low = qfluxb.getWindow()->getOffset();
  IntVector qfluxb_high = qfluxb.getWindow()->getData()->size() + qfluxb_low - IntVector(1, 1, 1);
  int qfluxb_low_x = qfluxb_low.x();
  int qfluxb_high_x = qfluxb_high.x();
  int qfluxb_low_y = qfluxb_low.y();
  int qfluxb_high_y = qfluxb_high.y();
  int qfluxb_low_z = qfluxb_low.z();
  int qfluxb_high_z = qfluxb_high.z();
  IntVector abskg_low = abskg.getWindow()->getOffset();
  IntVector abskg_high = abskg.getWindow()->getData()->size() + abskg_low - IntVector(1, 1, 1);
  int abskg_low_x = abskg_low.x();
  int abskg_high_x = abskg_high.x();
  int abskg_low_y = abskg_low.y();
  int abskg_high_y = abskg_high.y();
  int abskg_low_z = abskg_low.z();
  int abskg_high_z = abskg_high.z();
  IntVector shgamma_low = shgamma.getWindow()->getOffset();
  IntVector shgamma_high = shgamma.getWindow()->getData()->size() + shgamma_low - IntVector(1, 1, 1);
  int shgamma_low_x = shgamma_low.x();
  int shgamma_high_x = shgamma_high.x();
  int shgamma_low_y = shgamma_low.y();
  int shgamma_high_y = shgamma_high.y();
  int shgamma_low_z = shgamma_low.z();
  int shgamma_high_z = shgamma_high.z();
  IntVector esrcg_low = esrcg.getWindow()->getOffset();
  IntVector esrcg_high = esrcg.getWindow()->getData()->size() + esrcg_low - IntVector(1, 1, 1);
  int esrcg_low_x = esrcg_low.x();
  int esrcg_high_x = esrcg_high.x();
  int esrcg_low_y = esrcg_low.y();
  int esrcg_high_y = esrcg_high.y();
  int esrcg_low_z = esrcg_low.z();
  int esrcg_high_z = esrcg_high.z();
  IntVector src_low = src.getWindow()->getOffset();
  IntVector src_high = src.getWindow()->getData()->size() + src_low - IntVector(1, 1, 1);
  int src_low_x = src_low.x();
  int src_high_x = src_high.x();
  int src_low_y = src_low.y();
  int src_high_y = src_high.y();
  int src_low_z = src_low.z();
  int src_high_z = src_high.z();
  int fraction_low = fraction.low();
  int fraction_high = fraction.high();
  int fractiontwo_low = fractiontwo.low();
  int fractiontwo_high = fractiontwo.high();
  rshresults_(idxlo.get_pointer(),
idxhi.get_pointer(),
&cenint_low_x, &cenint_low_y, &cenint_low_z, &cenint_high_x, &cenint_high_y, &cenint_high_z, cenint.getPointer(),
&volq_low_x, &volq_low_y, &volq_low_z, &volq_high_x, &volq_high_y, &volq_high_z, volq.getPointer(),
&pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
&ffield,
&xx_low, &xx_high, xx.get_objs(),
&yy_low, &yy_high, yy.get_objs(),
&zz_low, &zz_high, zz.get_objs(),
&tg_low_x, &tg_low_y, &tg_low_z, &tg_high_x, &tg_high_y, &tg_high_z, tg.getPointer(),
&qfluxe_low_x, &qfluxe_low_y, &qfluxe_low_z, &qfluxe_high_x, &qfluxe_high_y, &qfluxe_high_z, qfluxe.getPointer(),
&qfluxw_low_x, &qfluxw_low_y, &qfluxw_low_z, &qfluxw_high_x, &qfluxw_high_y, &qfluxw_high_z, qfluxw.getPointer(),
&qfluxn_low_x, &qfluxn_low_y, &qfluxn_low_z, &qfluxn_high_x, &qfluxn_high_y, &qfluxn_high_z, qfluxn.getPointer(),
&qfluxs_low_x, &qfluxs_low_y, &qfluxs_low_z, &qfluxs_high_x, &qfluxs_high_y, &qfluxs_high_z, qfluxs.getPointer(),
&qfluxt_low_x, &qfluxt_low_y, &qfluxt_low_z, &qfluxt_high_x, &qfluxt_high_y, &qfluxt_high_z, qfluxt.getPointer(),
&qfluxb_low_x, &qfluxb_low_y, &qfluxb_low_z, &qfluxb_high_x, &qfluxb_high_y, &qfluxb_high_z, qfluxb.getPointer(),
&abskg_low_x, &abskg_low_y, &abskg_low_z, &abskg_high_x, &abskg_high_y, &abskg_high_z, abskg.getPointer(),
&shgamma_low_x, &shgamma_low_y, &shgamma_low_z, &shgamma_high_x, &shgamma_high_y, &shgamma_high_z, shgamma.getPointer(),
&esrcg_low_x, &esrcg_low_y, &esrcg_low_z, &esrcg_high_x, &esrcg_high_y, &esrcg_high_z, esrcg.getPointer(),
&src_low_x, &src_low_y, &src_low_z, &src_high_x, &src_high_y, &src_high_z, src.getPointer(),
&fraction_low, &fraction_high, fraction.get_objs(),
&fractiontwo_low, &fractiontwo_high, fractiontwo.get_objs(),
&bands);
}

#else /* !__cplusplus */
C Assuming this is fortran code

      subroutine rshresults(idxlo, idxhi, cenint_low_x, cenint_low_y, 
     & cenint_low_z, cenint_high_x, cenint_high_y, cenint_high_z, 
     & cenint, volq_low_x, volq_low_y, volq_low_z, volq_high_x, 
     & volq_high_y, volq_high_z, volq, pcell_low_x, pcell_low_y, 
     & pcell_low_z, pcell_high_x, pcell_high_y, pcell_high_z, pcell, 
     & ffield, xx_low, xx_high, xx, yy_low, yy_high, yy, zz_low, 
     & zz_high, zz, tg_low_x, tg_low_y, tg_low_z, tg_high_x, tg_high_y,
     &  tg_high_z, tg, qfluxe_low_x, qfluxe_low_y, qfluxe_low_z, 
     & qfluxe_high_x, qfluxe_high_y, qfluxe_high_z, qfluxe, 
     & qfluxw_low_x, qfluxw_low_y, qfluxw_low_z, qfluxw_high_x, 
     & qfluxw_high_y, qfluxw_high_z, qfluxw, qfluxn_low_x, qfluxn_low_y
     & , qfluxn_low_z, qfluxn_high_x, qfluxn_high_y, qfluxn_high_z, 
     & qfluxn, qfluxs_low_x, qfluxs_low_y, qfluxs_low_z, qfluxs_high_x,
     &  qfluxs_high_y, qfluxs_high_z, qfluxs, qfluxt_low_x, 
     & qfluxt_low_y, qfluxt_low_z, qfluxt_high_x, qfluxt_high_y, 
     & qfluxt_high_z, qfluxt, qfluxb_low_x, qfluxb_low_y, qfluxb_low_z,
     &  qfluxb_high_x, qfluxb_high_y, qfluxb_high_z, qfluxb, 
     & abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z, abskg, shgamma_low_x, shgamma_low_y,
     &  shgamma_low_z, shgamma_high_x, shgamma_high_y, shgamma_high_z, 
     & shgamma, esrcg_low_x, esrcg_low_y, esrcg_low_z, esrcg_high_x, 
     & esrcg_high_y, esrcg_high_z, esrcg, src_low_x, src_low_y, 
     & src_low_z, src_high_x, src_high_y, src_high_z, src, fraction_low
     & , fraction_high, fraction, fractiontwo_low, fractiontwo_high, 
     & fractiontwo, bands)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
      integer cenint_low_x, cenint_low_y, cenint_low_z, cenint_high_x, 
     & cenint_high_y, cenint_high_z
      double precision cenint(cenint_low_x:cenint_high_x, cenint_low_y:
     & cenint_high_y, cenint_low_z:cenint_high_z)
      integer volq_low_x, volq_low_y, volq_low_z, volq_high_x, 
     & volq_high_y, volq_high_z
      double precision volq(volq_low_x:volq_high_x, volq_low_y:
     & volq_high_y, volq_low_z:volq_high_z)
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer ffield
      integer xx_low
      integer xx_high
      double precision xx(xx_low:xx_high)
      integer yy_low
      integer yy_high
      double precision yy(yy_low:yy_high)
      integer zz_low
      integer zz_high
      double precision zz(zz_low:zz_high)
      integer tg_low_x, tg_low_y, tg_low_z, tg_high_x, tg_high_y, 
     & tg_high_z
      double precision tg(tg_low_x:tg_high_x, tg_low_y:tg_high_y, 
     & tg_low_z:tg_high_z)
      integer qfluxe_low_x, qfluxe_low_y, qfluxe_low_z, qfluxe_high_x, 
     & qfluxe_high_y, qfluxe_high_z
      double precision qfluxe(qfluxe_low_x:qfluxe_high_x, qfluxe_low_y:
     & qfluxe_high_y, qfluxe_low_z:qfluxe_high_z)
      integer qfluxw_low_x, qfluxw_low_y, qfluxw_low_z, qfluxw_high_x, 
     & qfluxw_high_y, qfluxw_high_z
      double precision qfluxw(qfluxw_low_x:qfluxw_high_x, qfluxw_low_y:
     & qfluxw_high_y, qfluxw_low_z:qfluxw_high_z)
      integer qfluxn_low_x, qfluxn_low_y, qfluxn_low_z, qfluxn_high_x, 
     & qfluxn_high_y, qfluxn_high_z
      double precision qfluxn(qfluxn_low_x:qfluxn_high_x, qfluxn_low_y:
     & qfluxn_high_y, qfluxn_low_z:qfluxn_high_z)
      integer qfluxs_low_x, qfluxs_low_y, qfluxs_low_z, qfluxs_high_x, 
     & qfluxs_high_y, qfluxs_high_z
      double precision qfluxs(qfluxs_low_x:qfluxs_high_x, qfluxs_low_y:
     & qfluxs_high_y, qfluxs_low_z:qfluxs_high_z)
      integer qfluxt_low_x, qfluxt_low_y, qfluxt_low_z, qfluxt_high_x, 
     & qfluxt_high_y, qfluxt_high_z
      double precision qfluxt(qfluxt_low_x:qfluxt_high_x, qfluxt_low_y:
     & qfluxt_high_y, qfluxt_low_z:qfluxt_high_z)
      integer qfluxb_low_x, qfluxb_low_y, qfluxb_low_z, qfluxb_high_x, 
     & qfluxb_high_y, qfluxb_high_z
      double precision qfluxb(qfluxb_low_x:qfluxb_high_x, qfluxb_low_y:
     & qfluxb_high_y, qfluxb_low_z:qfluxb_high_z)
      integer abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z
      double precision abskg(abskg_low_x:abskg_high_x, abskg_low_y:
     & abskg_high_y, abskg_low_z:abskg_high_z)
      integer shgamma_low_x, shgamma_low_y, shgamma_low_z, 
     & shgamma_high_x, shgamma_high_y, shgamma_high_z
      double precision shgamma(shgamma_low_x:shgamma_high_x, 
     & shgamma_low_y:shgamma_high_y, shgamma_low_z:shgamma_high_z)
      integer esrcg_low_x, esrcg_low_y, esrcg_low_z, esrcg_high_x, 
     & esrcg_high_y, esrcg_high_z
      double precision esrcg(esrcg_low_x:esrcg_high_x, esrcg_low_y:
     & esrcg_high_y, esrcg_low_z:esrcg_high_z)
      integer src_low_x, src_low_y, src_low_z, src_high_x, src_high_y, 
     & src_high_z
      double precision src(src_low_x:src_high_x, src_low_y:src_high_y, 
     & src_low_z:src_high_z)
      integer fraction_low
      integer fraction_high
      double precision fraction(fraction_low:fraction_high)
      integer fractiontwo_low
      integer fractiontwo_high
      double precision fractiontwo(fractiontwo_low:fractiontwo_high)
      integer bands
#endif /* __cplusplus */

#endif /* fspec_rshresults */

#ifndef PASS1
#define PASS1(x) x/**/_low, x/**/_high, x
#endif
#ifndef PASS3
#define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
