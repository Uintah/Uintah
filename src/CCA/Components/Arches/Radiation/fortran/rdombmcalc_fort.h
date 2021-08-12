
#ifndef fspec_rdombmcalc
#define fspec_rdombmcalc

#ifdef __cplusplus

#include <CCA/Components/Arches/Radiation/fortran/FortranNameMangle.h>

extern "C" void F_rdombmcalc(int* idxlo,
                           int* idxhi,
                           int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                           int* ffield,
                           int* xx_low, int* xx_high, double* xx_ptr,
                           int* zz_low, int* zz_high, double* zz_ptr,
                           int* sew_low, int* sew_high, double* sew_ptr,
                           int* sns_low, int* sns_high, double* sns_ptr,
                           int* stb_low, int* stb_high, double* stb_ptr,
                           int* volume_low_x, int* volume_low_y, int* volume_low_z, int* volume_high_x, int* volume_high_y, int* volume_high_z, double* volume_ptr,
                           double* areaew,
                           int* arean_low, int* arean_high, double* arean_ptr,
                           int* areatb_low, int* areatb_high, double* areatb_ptr,
                           int* srcbm_low, int* srcbm_high, double* srcbm_ptr,
                           int* qfluxbbm_low, int* qfluxbbm_high, double* qfluxbbm_ptr,
                           int* src_low_x, int* src_low_y, int* src_low_z, int* src_high_x, int* src_high_y, int* src_high_z, double* src_ptr,
                           int* qfluxe_low_x, int* qfluxe_low_y, int* qfluxe_low_z, int* qfluxe_high_x, int* qfluxe_high_y, int* qfluxe_high_z, double* qfluxe_ptr,
                           int* qfluxw_low_x, int* qfluxw_low_y, int* qfluxw_low_z, int* qfluxw_high_x, int* qfluxw_high_y, int* qfluxw_high_z, double* qfluxw_ptr,
                           int* qfluxn_low_x, int* qfluxn_low_y, int* qfluxn_low_z, int* qfluxn_high_x, int* qfluxn_high_y, int* qfluxn_high_z, double* qfluxn_ptr,
                           int* qfluxs_low_x, int* qfluxs_low_y, int* qfluxs_low_z, int* qfluxs_high_x, int* qfluxs_high_y, int* qfluxs_high_z, double* qfluxs_ptr,
                           int* qfluxt_low_x, int* qfluxt_low_y, int* qfluxt_low_z, int* qfluxt_high_x, int* qfluxt_high_y, int* qfluxt_high_z, double* qfluxt_ptr,
                           int* qfluxb_low_x, int* qfluxb_low_y, int* qfluxb_low_z, int* qfluxb_high_x, int* qfluxb_high_y, int* qfluxb_high_z, double* qfluxb_ptr,
                           bool* lprobone,
                           bool* lprobtwo,
                           bool* lprobthree,
                           int* srcpone_low, int* srcpone_high, double* srcpone_ptr,
                           int* volq_low_x, int* volq_low_y, int* volq_low_z, int* volq_high_x, int* volq_high_y, int* volq_high_z, double* volq_ptr,
                           double* srcsum);

static void fort_rdombmcalc( Uintah::IntVector & idxlo,
                             Uintah::IntVector & idxhi,
                             Uintah::constCCVariable<int> & pcell,
                             int & ffield,
                             Uintah::OffsetArray1<double> & xx,
                             Uintah::OffsetArray1<double> & zz,
                             Uintah::OffsetArray1<double> & sew,
                             Uintah::OffsetArray1<double> & sns,
                             Uintah::OffsetArray1<double> & stb,
                             Uintah::CCVariable<double> & volume,
                             double & areaew,
                             Uintah::OffsetArray1<double> & arean,
                             Uintah::OffsetArray1<double> & areatb,
                             Uintah::OffsetArray1<double> & srcbm,
                             Uintah::OffsetArray1<double> & qfluxbbm,
                             Uintah::CCVariable<double> & src,
                             Uintah::CCVariable<double> & qfluxe,
                             Uintah::CCVariable<double> & qfluxw,
                             Uintah::CCVariable<double> & qfluxn,
                             Uintah::CCVariable<double> & qfluxs,
                             Uintah::CCVariable<double> & qfluxt,
                             Uintah::CCVariable<double> & qfluxb,
                             bool & lprobone,
                             bool & lprobtwo,
                             bool & lprobthree,
                             Uintah::OffsetArray1<double> & srcpone,
                             Uintah::CCVariable<double> & volq,
                             double & srcsum )
{
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  int xx_low = xx.low();
  int xx_high = xx.high();
  int zz_low = zz.low();
  int zz_high = zz.high();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  Uintah::IntVector volume_low = volume.getWindow()->getOffset();
  Uintah::IntVector volume_high = volume.getWindow()->getData()->size() + volume_low - Uintah::IntVector(1, 1, 1);
  int volume_low_x = volume_low.x();
  int volume_high_x = volume_high.x();
  int volume_low_y = volume_low.y();
  int volume_high_y = volume_high.y();
  int volume_low_z = volume_low.z();
  int volume_high_z = volume_high.z();
  int arean_low = arean.low();
  int arean_high = arean.high();
  int areatb_low = areatb.low();
  int areatb_high = areatb.high();
  int srcbm_low = srcbm.low();
  int srcbm_high = srcbm.high();
  int qfluxbbm_low = qfluxbbm.low();
  int qfluxbbm_high = qfluxbbm.high();
  Uintah::IntVector src_low = src.getWindow()->getOffset();
  Uintah::IntVector src_high = src.getWindow()->getData()->size() + src_low - Uintah::IntVector(1, 1, 1);
  int src_low_x = src_low.x();
  int src_high_x = src_high.x();
  int src_low_y = src_low.y();
  int src_high_y = src_high.y();
  int src_low_z = src_low.z();
  int src_high_z = src_high.z();
  Uintah::IntVector qfluxe_low = qfluxe.getWindow()->getOffset();
  Uintah::IntVector qfluxe_high = qfluxe.getWindow()->getData()->size() + qfluxe_low - Uintah::IntVector(1, 1, 1);
  int qfluxe_low_x = qfluxe_low.x();
  int qfluxe_high_x = qfluxe_high.x();
  int qfluxe_low_y = qfluxe_low.y();
  int qfluxe_high_y = qfluxe_high.y();
  int qfluxe_low_z = qfluxe_low.z();
  int qfluxe_high_z = qfluxe_high.z();
  Uintah::IntVector qfluxw_low = qfluxw.getWindow()->getOffset();
  Uintah::IntVector qfluxw_high = qfluxw.getWindow()->getData()->size() + qfluxw_low - Uintah::IntVector(1, 1, 1);
  int qfluxw_low_x = qfluxw_low.x();
  int qfluxw_high_x = qfluxw_high.x();
  int qfluxw_low_y = qfluxw_low.y();
  int qfluxw_high_y = qfluxw_high.y();
  int qfluxw_low_z = qfluxw_low.z();
  int qfluxw_high_z = qfluxw_high.z();
  Uintah::IntVector qfluxn_low = qfluxn.getWindow()->getOffset();
  Uintah::IntVector qfluxn_high = qfluxn.getWindow()->getData()->size() + qfluxn_low - Uintah::IntVector(1, 1, 1);
  int qfluxn_low_x = qfluxn_low.x();
  int qfluxn_high_x = qfluxn_high.x();
  int qfluxn_low_y = qfluxn_low.y();
  int qfluxn_high_y = qfluxn_high.y();
  int qfluxn_low_z = qfluxn_low.z();
  int qfluxn_high_z = qfluxn_high.z();
  Uintah::IntVector qfluxs_low = qfluxs.getWindow()->getOffset();
  Uintah::IntVector qfluxs_high = qfluxs.getWindow()->getData()->size() + qfluxs_low - Uintah::IntVector(1, 1, 1);
  int qfluxs_low_x = qfluxs_low.x();
  int qfluxs_high_x = qfluxs_high.x();
  int qfluxs_low_y = qfluxs_low.y();
  int qfluxs_high_y = qfluxs_high.y();
  int qfluxs_low_z = qfluxs_low.z();
  int qfluxs_high_z = qfluxs_high.z();
  Uintah::IntVector qfluxt_low = qfluxt.getWindow()->getOffset();
  Uintah::IntVector qfluxt_high = qfluxt.getWindow()->getData()->size() + qfluxt_low - Uintah::IntVector(1, 1, 1);
  int qfluxt_low_x = qfluxt_low.x();
  int qfluxt_high_x = qfluxt_high.x();
  int qfluxt_low_y = qfluxt_low.y();
  int qfluxt_high_y = qfluxt_high.y();
  int qfluxt_low_z = qfluxt_low.z();
  int qfluxt_high_z = qfluxt_high.z();
  Uintah::IntVector qfluxb_low = qfluxb.getWindow()->getOffset();
  Uintah::IntVector qfluxb_high = qfluxb.getWindow()->getData()->size() + qfluxb_low - Uintah::IntVector(1, 1, 1);
  int qfluxb_low_x = qfluxb_low.x();
  int qfluxb_high_x = qfluxb_high.x();
  int qfluxb_low_y = qfluxb_low.y();
  int qfluxb_high_y = qfluxb_high.y();
  int qfluxb_low_z = qfluxb_low.z();
  int qfluxb_high_z = qfluxb_high.z();
  int srcpone_low = srcpone.low();
  int srcpone_high = srcpone.high();
  Uintah::IntVector volq_low = volq.getWindow()->getOffset();
  Uintah::IntVector volq_high = volq.getWindow()->getData()->size() + volq_low - Uintah::IntVector(1, 1, 1);
  int volq_low_x = volq_low.x();
  int volq_high_x = volq_high.x();
  int volq_low_y = volq_low.y();
  int volq_high_y = volq_high.y();
  int volq_low_z = volq_low.z();
  int volq_high_z = volq_high.z();
  F_rdombmcalc( idxlo.get_pointer(),
              idxhi.get_pointer(),
              &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
              &ffield,
              &xx_low, &xx_high, xx.get_objs(),
              &zz_low, &zz_high, zz.get_objs(),
              &sew_low, &sew_high, sew.get_objs(),
              &sns_low, &sns_high, sns.get_objs(),
              &stb_low, &stb_high, stb.get_objs(),
              &volume_low_x, &volume_low_y, &volume_low_z, &volume_high_x, &volume_high_y, &volume_high_z, volume.getPointer(),
              &areaew,
              &arean_low, &arean_high, arean.get_objs(),
              &areatb_low, &areatb_high, areatb.get_objs(),
              &srcbm_low, &srcbm_high, srcbm.get_objs(),
              &qfluxbbm_low, &qfluxbbm_high, qfluxbbm.get_objs(),
              &src_low_x, &src_low_y, &src_low_z, &src_high_x, &src_high_y, &src_high_z, src.getPointer(),
              &qfluxe_low_x, &qfluxe_low_y, &qfluxe_low_z, &qfluxe_high_x, &qfluxe_high_y, &qfluxe_high_z, qfluxe.getPointer(),
              &qfluxw_low_x, &qfluxw_low_y, &qfluxw_low_z, &qfluxw_high_x, &qfluxw_high_y, &qfluxw_high_z, qfluxw.getPointer(),
              &qfluxn_low_x, &qfluxn_low_y, &qfluxn_low_z, &qfluxn_high_x, &qfluxn_high_y, &qfluxn_high_z, qfluxn.getPointer(),
              &qfluxs_low_x, &qfluxs_low_y, &qfluxs_low_z, &qfluxs_high_x, &qfluxs_high_y, &qfluxs_high_z, qfluxs.getPointer(),
              &qfluxt_low_x, &qfluxt_low_y, &qfluxt_low_z, &qfluxt_high_x, &qfluxt_high_y, &qfluxt_high_z, qfluxt.getPointer(),
              &qfluxb_low_x, &qfluxb_low_y, &qfluxb_low_z, &qfluxb_high_x, &qfluxb_high_y, &qfluxb_high_z, qfluxb.getPointer(),
              &lprobone,
              &lprobtwo,
              &lprobthree,
              &srcpone_low, &srcpone_high, srcpone.get_objs(),
              &volq_low_x, &volq_low_y, &volq_low_z, &volq_high_x, &volq_high_y, &volq_high_z, volq.getPointer(),
              &srcsum );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine RDOMBMCALC(idxlo, idxhi, pcell_low_x, pcell_low_y,
     & pcell_low_z, pcell_high_x, pcell_high_y, pcell_high_z, pcell, 
     & ffield, xx_low, xx_high, xx, zz_low, zz_high, zz, sew_low, 
     & sew_high, sew, sns_low, sns_high, sns, stb_low, stb_high, stb, 
     & volume_low_x, volume_low_y, volume_low_z, volume_high_x, 
     & volume_high_y, volume_high_z, volume, areaew, arean_low, 
     & arean_high, arean, areatb_low, areatb_high, areatb, srcbm_low, 
     & srcbm_high, srcbm, qfluxbbm_low, qfluxbbm_high, qfluxbbm, 
     & src_low_x, src_low_y, src_low_z, src_high_x, src_high_y, 
     & src_high_z, src, qfluxe_low_x, qfluxe_low_y, qfluxe_low_z, 
     & qfluxe_high_x, qfluxe_high_y, qfluxe_high_z, qfluxe, 
     & qfluxw_low_x, qfluxw_low_y, qfluxw_low_z, qfluxw_high_x, 
     & qfluxw_high_y, qfluxw_high_z, qfluxw, qfluxn_low_x, qfluxn_low_y
     & , qfluxn_low_z, qfluxn_high_x, qfluxn_high_y, qfluxn_high_z, 
     & qfluxn, qfluxs_low_x, qfluxs_low_y, qfluxs_low_z, qfluxs_high_x,
     &  qfluxs_high_y, qfluxs_high_z, qfluxs, qfluxt_low_x, 
     & qfluxt_low_y, qfluxt_low_z, qfluxt_high_x, qfluxt_high_y, 
     & qfluxt_high_z, qfluxt, qfluxb_low_x, qfluxb_low_y, qfluxb_low_z,
     &  qfluxb_high_x, qfluxb_high_y, qfluxb_high_z, qfluxb, lprobone, 
     & lprobtwo, lprobthree, srcpone_low, srcpone_high, srcpone, 
     & volq_low_x, volq_low_y, volq_low_z, volq_high_x, volq_high_y, 
     & volq_high_z, volq, srcsum)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer ffield
      integer xx_low
      integer xx_high
      double precision xx(xx_low:xx_high)
      integer zz_low
      integer zz_high
      double precision zz(zz_low:zz_high)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer volume_low_x, volume_low_y, volume_low_z, volume_high_x, 
     & volume_high_y, volume_high_z
      double precision volume(volume_low_x:volume_high_x, volume_low_y:
     & volume_high_y, volume_low_z:volume_high_z)
      double precision areaew
      integer arean_low
      integer arean_high
      double precision arean(arean_low:arean_high)
      integer areatb_low
      integer areatb_high
      double precision areatb(areatb_low:areatb_high)
      integer srcbm_low
      integer srcbm_high
      double precision srcbm(srcbm_low:srcbm_high)
      integer qfluxbbm_low
      integer qfluxbbm_high
      double precision qfluxbbm(qfluxbbm_low:qfluxbbm_high)
      integer src_low_x, src_low_y, src_low_z, src_high_x, src_high_y, 
     & src_high_z
      double precision src(src_low_x:src_high_x, src_low_y:src_high_y, 
     & src_low_z:src_high_z)
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
      logical*1 lprobone
      logical*1 lprobtwo
      logical*1 lprobthree
      integer srcpone_low
      integer srcpone_high
      double precision srcpone(srcpone_low:srcpone_high)
      integer volq_low_x, volq_low_y, volq_low_z, volq_high_x, 
     & volq_high_y, volq_high_z
      double precision volq(volq_low_x:volq_high_x, volq_low_y:
     & volq_high_y, volq_low_z:volq_high_z)
      double precision srcsum
#endif /* __cplusplus */

#endif /* fspec_rdombmcalc */
