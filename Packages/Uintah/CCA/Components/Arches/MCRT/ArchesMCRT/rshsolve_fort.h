
#ifndef fspec_rshsolve
#define fspec_rshsolve

#ifdef __cplusplus

extern "C" void rshsolve_(int* idxlo,
int* idxhi,
int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
int* wall,
int* ffield,
int* sew_low, int* sew_high, double* sew_ptr,
int* sns_low, int* sns_high, double* sns_ptr,
int* stb_low, int* stb_high, double* stb_ptr,
int* xx_low, int* xx_high, double* xx_ptr,
int* yy_low, int* yy_high, double* yy_ptr,
int* zz_low, int* zz_high, double* zz_ptr,
int* esrcg_low_x, int* esrcg_low_y, int* esrcg_low_z, int* esrcg_high_x, int* esrcg_high_y, int* esrcg_high_z, double* esrcg_ptr,
int* tg_low_x, int* tg_low_y, int* tg_low_z, int* tg_high_x, int* tg_high_y, int* tg_high_z, double* tg_ptr,
int* abskg_low_x, int* abskg_low_y, int* abskg_low_z, int* abskg_high_x, int* abskg_high_y, int* abskg_high_z, double* abskg_ptr,
int* shgamma_low_x, int* shgamma_low_y, int* shgamma_low_z, int* shgamma_high_x, int* shgamma_high_y, int* shgamma_high_z, double* shgamma_ptr,
int* volume_low_x, int* volume_low_y, int* volume_low_z, int* volume_high_x, int* volume_high_y, int* volume_high_z, double* volume_ptr,
int* su_low_x, int* su_low_y, int* su_low_z, int* su_high_x, int* su_high_y, int* su_high_z, double* su_ptr,
int* ae_low_x, int* ae_low_y, int* ae_low_z, int* ae_high_x, int* ae_high_y, int* ae_high_z, double* ae_ptr,
int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
int* an_low_x, int* an_low_y, int* an_low_z, int* an_high_x, int* an_high_y, int* an_high_z, double* an_ptr,
int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
int* at_low_x, int* at_low_y, int* at_low_z, int* at_high_x, int* at_high_y, int* at_high_z, double* at_ptr,
int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
int* ap_low_x, int* ap_low_y, int* ap_low_z, int* ap_high_x, int* ap_high_y, int* ap_high_z, double* ap_ptr,
double* areaew,
int* arean_low, int* arean_high, double* arean_ptr,
int* areatb_low, int* areatb_high, double* areatb_ptr,
int* volq_low_x, int* volq_low_y, int* volq_low_z, int* volq_high_x, int* volq_high_y, int* volq_high_z, double* volq_ptr,
int* src_low_x, int* src_low_y, int* src_low_z, int* src_high_x, int* src_high_y, int* src_high_z, double* src_ptr,
bool* plusX,
bool* plusY,
bool* plusZ,
int* fraction_low, int* fraction_high, double* fraction_ptr,
int* fractiontwo_low, int* fractiontwo_high, double* fractiontwo_ptr,
int* bands);

static void fort_rshsolve(IntVector& idxlo,
IntVector& idxhi,
constCCVariable<int>& pcell,
int& wall,
int& ffield,
OffsetArray1<double>& sew,
OffsetArray1<double>& sns,
OffsetArray1<double>& stb,
OffsetArray1<double>& xx,
OffsetArray1<double>& yy,
OffsetArray1<double>& zz,
CCVariable<double>& esrcg,
CCVariable<double>& tg,
CCVariable<double>& abskg,
CCVariable<double>& shgamma,
CCVariable<double>& volume,
CCVariable<double>& su,
CCVariable<double>& ae,
CCVariable<double>& aw,
CCVariable<double>& an,
CCVariable<double>& as,
CCVariable<double>& at,
CCVariable<double>& ab,
CCVariable<double>& ap,
double& areaew,
OffsetArray1<double>& arean,
OffsetArray1<double>& areatb,
CCVariable<double>& volq,
CCVariable<double>& src,
bool& plusX,
bool& plusY,
bool& plusZ,
OffsetArray1<double>& fraction,
OffsetArray1<double>& fractiontwo,
int& bands)
{
  IntVector pcell_low = pcell.getWindow()->getOffset();
  IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - IntVector(1, 1, 1);
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
  int xx_low = xx.low();
  int xx_high = xx.high();
  int yy_low = yy.low();
  int yy_high = yy.high();
  int zz_low = zz.low();
  int zz_high = zz.high();
  IntVector esrcg_low = esrcg.getWindow()->getOffset();
  IntVector esrcg_high = esrcg.getWindow()->getData()->size() + esrcg_low - IntVector(1, 1, 1);
  int esrcg_low_x = esrcg_low.x();
  int esrcg_high_x = esrcg_high.x();
  int esrcg_low_y = esrcg_low.y();
  int esrcg_high_y = esrcg_high.y();
  int esrcg_low_z = esrcg_low.z();
  int esrcg_high_z = esrcg_high.z();
  IntVector tg_low = tg.getWindow()->getOffset();
  IntVector tg_high = tg.getWindow()->getData()->size() + tg_low - IntVector(1, 1, 1);
  int tg_low_x = tg_low.x();
  int tg_high_x = tg_high.x();
  int tg_low_y = tg_low.y();
  int tg_high_y = tg_high.y();
  int tg_low_z = tg_low.z();
  int tg_high_z = tg_high.z();
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
  IntVector volume_low = volume.getWindow()->getOffset();
  IntVector volume_high = volume.getWindow()->getData()->size() + volume_low - IntVector(1, 1, 1);
  int volume_low_x = volume_low.x();
  int volume_high_x = volume_high.x();
  int volume_low_y = volume_low.y();
  int volume_high_y = volume_high.y();
  int volume_low_z = volume_low.z();
  int volume_high_z = volume_high.z();
  IntVector su_low = su.getWindow()->getOffset();
  IntVector su_high = su.getWindow()->getData()->size() + su_low - IntVector(1, 1, 1);
  int su_low_x = su_low.x();
  int su_high_x = su_high.x();
  int su_low_y = su_low.y();
  int su_high_y = su_high.y();
  int su_low_z = su_low.z();
  int su_high_z = su_high.z();
  IntVector ae_low = ae.getWindow()->getOffset();
  IntVector ae_high = ae.getWindow()->getData()->size() + ae_low - IntVector(1, 1, 1);
  int ae_low_x = ae_low.x();
  int ae_high_x = ae_high.x();
  int ae_low_y = ae_low.y();
  int ae_high_y = ae_high.y();
  int ae_low_z = ae_low.z();
  int ae_high_z = ae_high.z();
  IntVector aw_low = aw.getWindow()->getOffset();
  IntVector aw_high = aw.getWindow()->getData()->size() + aw_low - IntVector(1, 1, 1);
  int aw_low_x = aw_low.x();
  int aw_high_x = aw_high.x();
  int aw_low_y = aw_low.y();
  int aw_high_y = aw_high.y();
  int aw_low_z = aw_low.z();
  int aw_high_z = aw_high.z();
  IntVector an_low = an.getWindow()->getOffset();
  IntVector an_high = an.getWindow()->getData()->size() + an_low - IntVector(1, 1, 1);
  int an_low_x = an_low.x();
  int an_high_x = an_high.x();
  int an_low_y = an_low.y();
  int an_high_y = an_high.y();
  int an_low_z = an_low.z();
  int an_high_z = an_high.z();
  IntVector as_low = as.getWindow()->getOffset();
  IntVector as_high = as.getWindow()->getData()->size() + as_low - IntVector(1, 1, 1);
  int as_low_x = as_low.x();
  int as_high_x = as_high.x();
  int as_low_y = as_low.y();
  int as_high_y = as_high.y();
  int as_low_z = as_low.z();
  int as_high_z = as_high.z();
  IntVector at_low = at.getWindow()->getOffset();
  IntVector at_high = at.getWindow()->getData()->size() + at_low - IntVector(1, 1, 1);
  int at_low_x = at_low.x();
  int at_high_x = at_high.x();
  int at_low_y = at_low.y();
  int at_high_y = at_high.y();
  int at_low_z = at_low.z();
  int at_high_z = at_high.z();
  IntVector ab_low = ab.getWindow()->getOffset();
  IntVector ab_high = ab.getWindow()->getData()->size() + ab_low - IntVector(1, 1, 1);
  int ab_low_x = ab_low.x();
  int ab_high_x = ab_high.x();
  int ab_low_y = ab_low.y();
  int ab_high_y = ab_high.y();
  int ab_low_z = ab_low.z();
  int ab_high_z = ab_high.z();
  IntVector ap_low = ap.getWindow()->getOffset();
  IntVector ap_high = ap.getWindow()->getData()->size() + ap_low - IntVector(1, 1, 1);
  int ap_low_x = ap_low.x();
  int ap_high_x = ap_high.x();
  int ap_low_y = ap_low.y();
  int ap_high_y = ap_high.y();
  int ap_low_z = ap_low.z();
  int ap_high_z = ap_high.z();
  int arean_low = arean.low();
  int arean_high = arean.high();
  int areatb_low = areatb.low();
  int areatb_high = areatb.high();
  IntVector volq_low = volq.getWindow()->getOffset();
  IntVector volq_high = volq.getWindow()->getData()->size() + volq_low - IntVector(1, 1, 1);
  int volq_low_x = volq_low.x();
  int volq_high_x = volq_high.x();
  int volq_low_y = volq_low.y();
  int volq_high_y = volq_high.y();
  int volq_low_z = volq_low.z();
  int volq_high_z = volq_high.z();
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
  rshsolve_(idxlo.get_pointer(),
idxhi.get_pointer(),
&pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
&wall,
&ffield,
&sew_low, &sew_high, sew.get_objs(),
&sns_low, &sns_high, sns.get_objs(),
&stb_low, &stb_high, stb.get_objs(),
&xx_low, &xx_high, xx.get_objs(),
&yy_low, &yy_high, yy.get_objs(),
&zz_low, &zz_high, zz.get_objs(),
&esrcg_low_x, &esrcg_low_y, &esrcg_low_z, &esrcg_high_x, &esrcg_high_y, &esrcg_high_z, esrcg.getPointer(),
&tg_low_x, &tg_low_y, &tg_low_z, &tg_high_x, &tg_high_y, &tg_high_z, tg.getPointer(),
&abskg_low_x, &abskg_low_y, &abskg_low_z, &abskg_high_x, &abskg_high_y, &abskg_high_z, abskg.getPointer(),
&shgamma_low_x, &shgamma_low_y, &shgamma_low_z, &shgamma_high_x, &shgamma_high_y, &shgamma_high_z, shgamma.getPointer(),
&volume_low_x, &volume_low_y, &volume_low_z, &volume_high_x, &volume_high_y, &volume_high_z, volume.getPointer(),
&su_low_x, &su_low_y, &su_low_z, &su_high_x, &su_high_y, &su_high_z, su.getPointer(),
&ae_low_x, &ae_low_y, &ae_low_z, &ae_high_x, &ae_high_y, &ae_high_z, ae.getPointer(),
&aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
&an_low_x, &an_low_y, &an_low_z, &an_high_x, &an_high_y, &an_high_z, an.getPointer(),
&as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
&at_low_x, &at_low_y, &at_low_z, &at_high_x, &at_high_y, &at_high_z, at.getPointer(),
&ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
&ap_low_x, &ap_low_y, &ap_low_z, &ap_high_x, &ap_high_y, &ap_high_z, ap.getPointer(),
&areaew,
&arean_low, &arean_high, arean.get_objs(),
&areatb_low, &areatb_high, areatb.get_objs(),
&volq_low_x, &volq_low_y, &volq_low_z, &volq_high_x, &volq_high_y, &volq_high_z, volq.getPointer(),
&src_low_x, &src_low_y, &src_low_z, &src_high_x, &src_high_y, &src_high_z, src.getPointer(),
&plusX,
&plusY,
&plusZ,
&fraction_low, &fraction_high, fraction.get_objs(),
&fractiontwo_low, &fractiontwo_high, fractiontwo.get_objs(),
&bands);
}

#else /* !__cplusplus */
C Assuming this is fortran code

      subroutine rshsolve(idxlo, idxhi, pcell_low_x, pcell_low_y, 
     & pcell_low_z, pcell_high_x, pcell_high_y, pcell_high_z, pcell, 
     & wall, ffield, sew_low, sew_high, sew, sns_low, sns_high, sns, 
     & stb_low, stb_high, stb, xx_low, xx_high, xx, yy_low, yy_high, yy
     & , zz_low, zz_high, zz, esrcg_low_x, esrcg_low_y, esrcg_low_z, 
     & esrcg_high_x, esrcg_high_y, esrcg_high_z, esrcg, tg_low_x, 
     & tg_low_y, tg_low_z, tg_high_x, tg_high_y, tg_high_z, tg, 
     & abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z, abskg, shgamma_low_x, shgamma_low_y,
     &  shgamma_low_z, shgamma_high_x, shgamma_high_y, shgamma_high_z, 
     & shgamma, volume_low_x, volume_low_y, volume_low_z, volume_high_x
     & , volume_high_y, volume_high_z, volume, su_low_x, su_low_y, 
     & su_low_z, su_high_x, su_high_y, su_high_z, su, ae_low_x, 
     & ae_low_y, ae_low_z, ae_high_x, ae_high_y, ae_high_z, ae, 
     & aw_low_x, aw_low_y, aw_low_z, aw_high_x, aw_high_y, aw_high_z, 
     & aw, an_low_x, an_low_y, an_low_z, an_high_x, an_high_y, 
     & an_high_z, an, as_low_x, as_low_y, as_low_z, as_high_x, 
     & as_high_y, as_high_z, as, at_low_x, at_low_y, at_low_z, 
     & at_high_x, at_high_y, at_high_z, at, ab_low_x, ab_low_y, 
     & ab_low_z, ab_high_x, ab_high_y, ab_high_z, ab, ap_low_x, 
     & ap_low_y, ap_low_z, ap_high_x, ap_high_y, ap_high_z, ap, areaew,
     &  arean_low, arean_high, arean, areatb_low, areatb_high, areatb, 
     & volq_low_x, volq_low_y, volq_low_z, volq_high_x, volq_high_y, 
     & volq_high_z, volq, src_low_x, src_low_y, src_low_z, src_high_x, 
     & src_high_y, src_high_z, src, plusX, plusY, plusZ, fraction_low, 
     & fraction_high, fraction, fractiontwo_low, fractiontwo_high, 
     & fractiontwo, bands)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer wall
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
      integer xx_low
      integer xx_high
      double precision xx(xx_low:xx_high)
      integer yy_low
      integer yy_high
      double precision yy(yy_low:yy_high)
      integer zz_low
      integer zz_high
      double precision zz(zz_low:zz_high)
      integer esrcg_low_x, esrcg_low_y, esrcg_low_z, esrcg_high_x, 
     & esrcg_high_y, esrcg_high_z
      double precision esrcg(esrcg_low_x:esrcg_high_x, esrcg_low_y:
     & esrcg_high_y, esrcg_low_z:esrcg_high_z)
      integer tg_low_x, tg_low_y, tg_low_z, tg_high_x, tg_high_y, 
     & tg_high_z
      double precision tg(tg_low_x:tg_high_x, tg_low_y:tg_high_y, 
     & tg_low_z:tg_high_z)
      integer abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z
      double precision abskg(abskg_low_x:abskg_high_x, abskg_low_y:
     & abskg_high_y, abskg_low_z:abskg_high_z)
      integer shgamma_low_x, shgamma_low_y, shgamma_low_z, 
     & shgamma_high_x, shgamma_high_y, shgamma_high_z
      double precision shgamma(shgamma_low_x:shgamma_high_x, 
     & shgamma_low_y:shgamma_high_y, shgamma_low_z:shgamma_high_z)
      integer volume_low_x, volume_low_y, volume_low_z, volume_high_x, 
     & volume_high_y, volume_high_z
      double precision volume(volume_low_x:volume_high_x, volume_low_y:
     & volume_high_y, volume_low_z:volume_high_z)
      integer su_low_x, su_low_y, su_low_z, su_high_x, su_high_y, 
     & su_high_z
      double precision su(su_low_x:su_high_x, su_low_y:su_high_y, 
     & su_low_z:su_high_z)
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
      double precision areaew
      integer arean_low
      integer arean_high
      double precision arean(arean_low:arean_high)
      integer areatb_low
      integer areatb_high
      double precision areatb(areatb_low:areatb_high)
      integer volq_low_x, volq_low_y, volq_low_z, volq_high_x, 
     & volq_high_y, volq_high_z
      double precision volq(volq_low_x:volq_high_x, volq_low_y:
     & volq_high_y, volq_low_z:volq_high_z)
      integer src_low_x, src_low_y, src_low_z, src_high_x, src_high_y, 
     & src_high_z
      double precision src(src_low_x:src_high_x, src_low_y:src_high_y, 
     & src_low_z:src_high_z)
      logical*1 plusX
      logical*1 plusY
      logical*1 plusZ
      integer fraction_low
      integer fraction_high
      double precision fraction(fraction_low:fraction_high)
      integer fractiontwo_low
      integer fractiontwo_high
      double precision fractiontwo(fractiontwo_low:fractiontwo_high)
      integer bands
#endif /* __cplusplus */

#endif /* fspec_rshsolve */

#ifndef PASS1
#define PASS1(x) x/**/_low, x/**/_high, x
#endif
#ifndef PASS3
#define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
