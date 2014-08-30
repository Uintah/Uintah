
#ifndef fspec_taucal_cc
#define fspec_taucal_cc

#ifdef __cplusplus

extern "C" void taucal_cc_(int* su_fcy_low_x, int* su_fcy_low_y, int* su_fcy_low_z, int* su_fcy_high_x, int* su_fcy_high_y, int* su_fcy_high_z, double* su_fcy_ptr,
                           int* sp_fcy_low_x, int* sp_fcy_low_y, int* sp_fcy_low_z, int* sp_fcy_high_x, int* sp_fcy_high_y, int* sp_fcy_high_z, double* sp_fcy_ptr,
                           int* dfx_fcy_low_x, int* dfx_fcy_low_y, int* dfx_fcy_low_z, int* dfx_fcy_high_x, int* dfx_fcy_high_y, int* dfx_fcy_high_z, double* dfx_fcy_ptr,
                           int* kstabu_low_x, int* kstabu_low_y, int* kstabu_low_z, int* kstabu_high_x, int* kstabu_high_y, int* kstabu_high_z, double* kstabu_ptr,
                           int* ug_cc_low_x, int* ug_cc_low_y, int* ug_cc_low_z, int* ug_cc_high_x, int* ug_cc_high_y, int* ug_cc_high_z, double* ug_cc_ptr,
                           int* up_fcy_low_x, int* up_fcy_low_y, int* up_fcy_low_z, int* up_fcy_high_x, int* up_fcy_high_y, int* up_fcy_high_z, double* up_fcy_ptr,
                           int* epsg_low_x, int* epsg_low_y, int* epsg_low_z, int* epsg_high_x, int* epsg_high_y, int* epsg_high_z, double* epsg_ptr,
                           int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                           int* dmicr_low_x, int* dmicr_low_y, int* dmicr_low_z, int* dmicr_high_x, int* dmicr_high_y, int* dmicr_high_z, double* dmicr_ptr,
                           int* epss_low_x, int* epss_low_y, int* epss_low_z, int* epss_high_x, int* epss_high_y, int* epss_high_z, double* epss_ptr,
                           int* sew_low, int* sew_high, double* sew_ptr,
                           int* stb_low, int* stb_high, double* stb_ptr,
                           int* yv_low, int* yv_high, double* yv_ptr,
                           int* yy_low, int* yy_high, double* yy_ptr,
                           double* viscos,
                           double* csmag,
                           double* sign,
                           int* idist,
                           int* idf,
                           int* idt1,
                           int* idt2,
                           int* i,
                           int* j,
                           int* k,
                           int* iorig,
                           int* jorig,
                           int* korig,
                           int* ioff,
                           int* joff,
                           int* koff,
                           int* indexflo,
                           int* indext1,
                           int* indext2,
                           bool* lmltm,
                           int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                           int* ffield);

static void fort_taucal_cc( Uintah::Array3<double> & su_fcy,
                            Uintah::Array3<double> & sp_fcy,
                            Uintah::Array3<double> & dfx_fcy,
                            Uintah::Array3<double> & kstabu,
                            Uintah::constCCVariable<double> & ug_cc,
                            Uintah::Array3<double> & up_fcy,
                            Uintah::constCCVariable<double> & epsg,
                            Uintah::constCCVariable<double> & den,
                            Uintah::constCCVariable<double> & dmicr,
                            Uintah::constCCVariable<double> & epss,
                            Uintah::OffsetArray1<double> & sew,
                            Uintah::OffsetArray1<double> & stb,
                            Uintah::OffsetArray1<double> & yv,
                            Uintah::OffsetArray1<double> & yy,
                            double & viscos,
                            double & csmag,
                            double & sign,
                            int & idist,
                            int & idf,
                            int & idt1,
                            int & idt2,
                            int & i,
                            int & j,
                            int & k,
                            int & iorig,
                            int & jorig,
                            int & korig,
                            int & ioff,
                            int & joff,
                            int & koff,
                            int & indexflo,
                            int & indext1,
                            int & indext2,
                            bool & lmltm,
                            Uintah::constCCVariable<int> & pcell,
                            int & ffield )
{
  Uintah::IntVector su_fcy_low = su_fcy.getWindow()->getOffset();
  Uintah::IntVector su_fcy_high = su_fcy.getWindow()->getData()->size() + su_fcy_low - Uintah::IntVector(1, 1, 1);
  int su_fcy_low_x = su_fcy_low.x();
  int su_fcy_high_x = su_fcy_high.x();
  int su_fcy_low_y = su_fcy_low.y();
  int su_fcy_high_y = su_fcy_high.y();
  int su_fcy_low_z = su_fcy_low.z();
  int su_fcy_high_z = su_fcy_high.z();
  Uintah::IntVector sp_fcy_low = sp_fcy.getWindow()->getOffset();
  Uintah::IntVector sp_fcy_high = sp_fcy.getWindow()->getData()->size() + sp_fcy_low - Uintah::IntVector(1, 1, 1);
  int sp_fcy_low_x = sp_fcy_low.x();
  int sp_fcy_high_x = sp_fcy_high.x();
  int sp_fcy_low_y = sp_fcy_low.y();
  int sp_fcy_high_y = sp_fcy_high.y();
  int sp_fcy_low_z = sp_fcy_low.z();
  int sp_fcy_high_z = sp_fcy_high.z();
  Uintah::IntVector dfx_fcy_low = dfx_fcy.getWindow()->getOffset();
  Uintah::IntVector dfx_fcy_high = dfx_fcy.getWindow()->getData()->size() + dfx_fcy_low - Uintah::IntVector(1, 1, 1);
  int dfx_fcy_low_x = dfx_fcy_low.x();
  int dfx_fcy_high_x = dfx_fcy_high.x();
  int dfx_fcy_low_y = dfx_fcy_low.y();
  int dfx_fcy_high_y = dfx_fcy_high.y();
  int dfx_fcy_low_z = dfx_fcy_low.z();
  int dfx_fcy_high_z = dfx_fcy_high.z();
  Uintah::IntVector kstabu_low = kstabu.getWindow()->getOffset();
  Uintah::IntVector kstabu_high = kstabu.getWindow()->getData()->size() + kstabu_low - Uintah::IntVector(1, 1, 1);
  int kstabu_low_x = kstabu_low.x();
  int kstabu_high_x = kstabu_high.x();
  int kstabu_low_y = kstabu_low.y();
  int kstabu_high_y = kstabu_high.y();
  int kstabu_low_z = kstabu_low.z();
  int kstabu_high_z = kstabu_high.z();
  Uintah::IntVector ug_cc_low = ug_cc.getWindow()->getOffset();
  Uintah::IntVector ug_cc_high = ug_cc.getWindow()->getData()->size() + ug_cc_low - Uintah::IntVector(1, 1, 1);
  int ug_cc_low_x = ug_cc_low.x();
  int ug_cc_high_x = ug_cc_high.x();
  int ug_cc_low_y = ug_cc_low.y();
  int ug_cc_high_y = ug_cc_high.y();
  int ug_cc_low_z = ug_cc_low.z();
  int ug_cc_high_z = ug_cc_high.z();
  Uintah::IntVector up_fcy_low = up_fcy.getWindow()->getOffset();
  Uintah::IntVector up_fcy_high = up_fcy.getWindow()->getData()->size() + up_fcy_low - Uintah::IntVector(1, 1, 1);
  int up_fcy_low_x = up_fcy_low.x();
  int up_fcy_high_x = up_fcy_high.x();
  int up_fcy_low_y = up_fcy_low.y();
  int up_fcy_high_y = up_fcy_high.y();
  int up_fcy_low_z = up_fcy_low.z();
  int up_fcy_high_z = up_fcy_high.z();
  Uintah::IntVector epsg_low = epsg.getWindow()->getOffset();
  Uintah::IntVector epsg_high = epsg.getWindow()->getData()->size() + epsg_low - Uintah::IntVector(1, 1, 1);
  int epsg_low_x = epsg_low.x();
  int epsg_high_x = epsg_high.x();
  int epsg_low_y = epsg_low.y();
  int epsg_high_y = epsg_high.y();
  int epsg_low_z = epsg_low.z();
  int epsg_high_z = epsg_high.z();
  Uintah::IntVector den_low = den.getWindow()->getOffset();
  Uintah::IntVector den_high = den.getWindow()->getData()->size() + den_low - Uintah::IntVector(1, 1, 1);
  int den_low_x = den_low.x();
  int den_high_x = den_high.x();
  int den_low_y = den_low.y();
  int den_high_y = den_high.y();
  int den_low_z = den_low.z();
  int den_high_z = den_high.z();
  Uintah::IntVector dmicr_low = dmicr.getWindow()->getOffset();
  Uintah::IntVector dmicr_high = dmicr.getWindow()->getData()->size() + dmicr_low - Uintah::IntVector(1, 1, 1);
  int dmicr_low_x = dmicr_low.x();
  int dmicr_high_x = dmicr_high.x();
  int dmicr_low_y = dmicr_low.y();
  int dmicr_high_y = dmicr_high.y();
  int dmicr_low_z = dmicr_low.z();
  int dmicr_high_z = dmicr_high.z();
  Uintah::IntVector epss_low = epss.getWindow()->getOffset();
  Uintah::IntVector epss_high = epss.getWindow()->getData()->size() + epss_low - Uintah::IntVector(1, 1, 1);
  int epss_low_x = epss_low.x();
  int epss_high_x = epss_high.x();
  int epss_low_y = epss_low.y();
  int epss_high_y = epss_high.y();
  int epss_low_z = epss_low.z();
  int epss_high_z = epss_high.z();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  int yv_low = yv.low();
  int yv_high = yv.high();
  int yy_low = yy.low();
  int yy_high = yy.high();
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  taucal_cc_( &su_fcy_low_x, &su_fcy_low_y, &su_fcy_low_z, &su_fcy_high_x, &su_fcy_high_y, &su_fcy_high_z, su_fcy.getPointer(),
              &sp_fcy_low_x, &sp_fcy_low_y, &sp_fcy_low_z, &sp_fcy_high_x, &sp_fcy_high_y, &sp_fcy_high_z, sp_fcy.getPointer(),
              &dfx_fcy_low_x, &dfx_fcy_low_y, &dfx_fcy_low_z, &dfx_fcy_high_x, &dfx_fcy_high_y, &dfx_fcy_high_z, dfx_fcy.getPointer(),
              &kstabu_low_x, &kstabu_low_y, &kstabu_low_z, &kstabu_high_x, &kstabu_high_y, &kstabu_high_z, kstabu.getPointer(),
              &ug_cc_low_x, &ug_cc_low_y, &ug_cc_low_z, &ug_cc_high_x, &ug_cc_high_y, &ug_cc_high_z, const_cast<double*>(ug_cc.getPointer()),
              &up_fcy_low_x, &up_fcy_low_y, &up_fcy_low_z, &up_fcy_high_x, &up_fcy_high_y, &up_fcy_high_z, up_fcy.getPointer(),
              &epsg_low_x, &epsg_low_y, &epsg_low_z, &epsg_high_x, &epsg_high_y, &epsg_high_z, const_cast<double*>(epsg.getPointer()),
              &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
              &dmicr_low_x, &dmicr_low_y, &dmicr_low_z, &dmicr_high_x, &dmicr_high_y, &dmicr_high_z, const_cast<double*>(dmicr.getPointer()),
              &epss_low_x, &epss_low_y, &epss_low_z, &epss_high_x, &epss_high_y, &epss_high_z, const_cast<double*>(epss.getPointer()),
              &sew_low, &sew_high, sew.get_objs(),
              &stb_low, &stb_high, stb.get_objs(),
              &yv_low, &yv_high, yv.get_objs(),
              &yy_low, &yy_high, yy.get_objs(),
              &viscos,
              &csmag,
              &sign,
              &idist,
              &idf,
              &idt1,
              &idt2,
              &i,
              &j,
              &k,
              &iorig,
              &jorig,
              &korig,
              &ioff,
              &joff,
              &koff,
              &indexflo,
              &indext1,
              &indext2,
              &lmltm,
              &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
              &ffield );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine taucal_cc(su_fcy_low_x, su_fcy_low_y, su_fcy_low_z, 
     & su_fcy_high_x, su_fcy_high_y, su_fcy_high_z, su_fcy, 
     & sp_fcy_low_x, sp_fcy_low_y, sp_fcy_low_z, sp_fcy_high_x, 
     & sp_fcy_high_y, sp_fcy_high_z, sp_fcy, dfx_fcy_low_x, 
     & dfx_fcy_low_y, dfx_fcy_low_z, dfx_fcy_high_x, dfx_fcy_high_y, 
     & dfx_fcy_high_z, dfx_fcy, kstabu_low_x, kstabu_low_y, 
     & kstabu_low_z, kstabu_high_x, kstabu_high_y, kstabu_high_z, 
     & kstabu, ug_cc_low_x, ug_cc_low_y, ug_cc_low_z, ug_cc_high_x, 
     & ug_cc_high_y, ug_cc_high_z, ug_cc, up_fcy_low_x, up_fcy_low_y, 
     & up_fcy_low_z, up_fcy_high_x, up_fcy_high_y, up_fcy_high_z, 
     & up_fcy, epsg_low_x, epsg_low_y, epsg_low_z, epsg_high_x, 
     & epsg_high_y, epsg_high_z, epsg, den_low_x, den_low_y, den_low_z,
     &  den_high_x, den_high_y, den_high_z, den, dmicr_low_x, 
     & dmicr_low_y, dmicr_low_z, dmicr_high_x, dmicr_high_y, 
     & dmicr_high_z, dmicr, epss_low_x, epss_low_y, epss_low_z, 
     & epss_high_x, epss_high_y, epss_high_z, epss, sew_low, sew_high, 
     & sew, stb_low, stb_high, stb, yv_low, yv_high, yv, yy_low, 
     & yy_high, yy, viscos, csmag, sign, idist, idf, idt1, idt2, i, j, 
     & k, iorig, jorig, korig, ioff, joff, koff, indexflo, indext1, 
     & indext2, lmltm, pcell_low_x, pcell_low_y, pcell_low_z, 
     & pcell_high_x, pcell_high_y, pcell_high_z, pcell, ffield)

      implicit none
      integer su_fcy_low_x, su_fcy_low_y, su_fcy_low_z, su_fcy_high_x, 
     & su_fcy_high_y, su_fcy_high_z
      double precision su_fcy(su_fcy_low_x:su_fcy_high_x, su_fcy_low_y:
     & su_fcy_high_y, su_fcy_low_z:su_fcy_high_z)
      integer sp_fcy_low_x, sp_fcy_low_y, sp_fcy_low_z, sp_fcy_high_x, 
     & sp_fcy_high_y, sp_fcy_high_z
      double precision sp_fcy(sp_fcy_low_x:sp_fcy_high_x, sp_fcy_low_y:
     & sp_fcy_high_y, sp_fcy_low_z:sp_fcy_high_z)
      integer dfx_fcy_low_x, dfx_fcy_low_y, dfx_fcy_low_z, 
     & dfx_fcy_high_x, dfx_fcy_high_y, dfx_fcy_high_z
      double precision dfx_fcy(dfx_fcy_low_x:dfx_fcy_high_x, 
     & dfx_fcy_low_y:dfx_fcy_high_y, dfx_fcy_low_z:dfx_fcy_high_z)
      integer kstabu_low_x, kstabu_low_y, kstabu_low_z, kstabu_high_x, 
     & kstabu_high_y, kstabu_high_z
      double precision kstabu(kstabu_low_x:kstabu_high_x, kstabu_low_y:
     & kstabu_high_y, kstabu_low_z:kstabu_high_z)
      integer ug_cc_low_x, ug_cc_low_y, ug_cc_low_z, ug_cc_high_x, 
     & ug_cc_high_y, ug_cc_high_z
      double precision ug_cc(ug_cc_low_x:ug_cc_high_x, ug_cc_low_y:
     & ug_cc_high_y, ug_cc_low_z:ug_cc_high_z)
      integer up_fcy_low_x, up_fcy_low_y, up_fcy_low_z, up_fcy_high_x, 
     & up_fcy_high_y, up_fcy_high_z
      double precision up_fcy(up_fcy_low_x:up_fcy_high_x, up_fcy_low_y:
     & up_fcy_high_y, up_fcy_low_z:up_fcy_high_z)
      integer epsg_low_x, epsg_low_y, epsg_low_z, epsg_high_x, 
     & epsg_high_y, epsg_high_z
      double precision epsg(epsg_low_x:epsg_high_x, epsg_low_y:
     & epsg_high_y, epsg_low_z:epsg_high_z)
      integer den_low_x, den_low_y, den_low_z, den_high_x, den_high_y, 
     & den_high_z
      double precision den(den_low_x:den_high_x, den_low_y:den_high_y, 
     & den_low_z:den_high_z)
      integer dmicr_low_x, dmicr_low_y, dmicr_low_z, dmicr_high_x, 
     & dmicr_high_y, dmicr_high_z
      double precision dmicr(dmicr_low_x:dmicr_high_x, dmicr_low_y:
     & dmicr_high_y, dmicr_low_z:dmicr_high_z)
      integer epss_low_x, epss_low_y, epss_low_z, epss_high_x, 
     & epss_high_y, epss_high_z
      double precision epss(epss_low_x:epss_high_x, epss_low_y:
     & epss_high_y, epss_low_z:epss_high_z)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer yv_low
      integer yv_high
      double precision yv(yv_low:yv_high)
      integer yy_low
      integer yy_high
      double precision yy(yy_low:yy_high)
      double precision viscos
      double precision csmag
      double precision sign
      integer idist
      integer idf
      integer idt1
      integer idt2
      integer i
      integer j
      integer k
      integer iorig
      integer jorig
      integer korig
      integer ioff
      integer joff
      integer koff
      integer indexflo
      integer indext1
      integer indext2
      logical*1 lmltm
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer ffield
#endif /* __cplusplus */

#endif /* fspec_taucal_cc */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
