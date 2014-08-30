
#ifndef fspec_rdomflux
#define fspec_rdomflux

#ifdef __cplusplus

extern "C" void rdomflux_(int* idxlo,
                          int* idxhi,
                          int* l,
                          int* oxi_low, int* oxi_high, double* oxi_ptr,
                          int* omu_low, int* omu_high, double* omu_ptr,
                          int* oeta_low, int* oeta_high, double* oeta_ptr,
                          int* wt_low, int* wt_high, double* wt_ptr,
                          int* cenint_low_x, int* cenint_low_y, int* cenint_low_z, int* cenint_high_x, int* cenint_high_y, int* cenint_high_z, double* cenint_ptr,
                          bool* plusX,
                          bool* plusY,
                          bool* plusZ,
                          int* qfluxe_low_x, int* qfluxe_low_y, int* qfluxe_low_z, int* qfluxe_high_x, int* qfluxe_high_y, int* qfluxe_high_z, double* qfluxe_ptr,
                          int* qfluxw_low_x, int* qfluxw_low_y, int* qfluxw_low_z, int* qfluxw_high_x, int* qfluxw_high_y, int* qfluxw_high_z, double* qfluxw_ptr,
                          int* qfluxn_low_x, int* qfluxn_low_y, int* qfluxn_low_z, int* qfluxn_high_x, int* qfluxn_high_y, int* qfluxn_high_z, double* qfluxn_ptr,
                          int* qfluxs_low_x, int* qfluxs_low_y, int* qfluxs_low_z, int* qfluxs_high_x, int* qfluxs_high_y, int* qfluxs_high_z, double* qfluxs_ptr,
                          int* qfluxt_low_x, int* qfluxt_low_y, int* qfluxt_low_z, int* qfluxt_high_x, int* qfluxt_high_y, int* qfluxt_high_z, double* qfluxt_ptr,
                          int* qfluxb_low_x, int* qfluxb_low_y, int* qfluxb_low_z, int* qfluxb_high_x, int* qfluxb_high_y, int* qfluxb_high_z, double* qfluxb_ptr);

static void fort_rdomflux( Uintah::IntVector & idxlo,
                           Uintah::IntVector & idxhi,
                           int & l,
                           Uintah::OffsetArray1<double> & oxi,
                           Uintah::OffsetArray1<double> & omu,
                           Uintah::OffsetArray1<double> & oeta,
                           Uintah::OffsetArray1<double> & wt,
                           Uintah::CCVariable<double> & cenint,
                           bool & plusX,
                           bool & plusY,
                           bool & plusZ,
                           Uintah::CCVariable<double> & qfluxe,
                           Uintah::CCVariable<double> & qfluxw,
                           Uintah::CCVariable<double> & qfluxn,
                           Uintah::CCVariable<double> & qfluxs,
                           Uintah::CCVariable<double> & qfluxt,
                           Uintah::CCVariable<double> & qfluxb )
{
  int oxi_low = oxi.low();
  int oxi_high = oxi.high();
  int omu_low = omu.low();
  int omu_high = omu.high();
  int oeta_low = oeta.low();
  int oeta_high = oeta.high();
  int wt_low = wt.low();
  int wt_high = wt.high();
  Uintah::IntVector cenint_low = cenint.getWindow()->getOffset();
  Uintah::IntVector cenint_high = cenint.getWindow()->getData()->size() + cenint_low - Uintah::IntVector(1, 1, 1);
  int cenint_low_x = cenint_low.x();
  int cenint_high_x = cenint_high.x();
  int cenint_low_y = cenint_low.y();
  int cenint_high_y = cenint_high.y();
  int cenint_low_z = cenint_low.z();
  int cenint_high_z = cenint_high.z();
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
  rdomflux_( idxlo.get_pointer(),
             idxhi.get_pointer(),
             &l,
             &oxi_low, &oxi_high, oxi.get_objs(),
             &omu_low, &omu_high, omu.get_objs(),
             &oeta_low, &oeta_high, oeta.get_objs(),
             &wt_low, &wt_high, wt.get_objs(),
             &cenint_low_x, &cenint_low_y, &cenint_low_z, &cenint_high_x, &cenint_high_y, &cenint_high_z, cenint.getPointer(),
             &plusX,
             &plusY,
             &plusZ,
             &qfluxe_low_x, &qfluxe_low_y, &qfluxe_low_z, &qfluxe_high_x, &qfluxe_high_y, &qfluxe_high_z, qfluxe.getPointer(),
             &qfluxw_low_x, &qfluxw_low_y, &qfluxw_low_z, &qfluxw_high_x, &qfluxw_high_y, &qfluxw_high_z, qfluxw.getPointer(),
             &qfluxn_low_x, &qfluxn_low_y, &qfluxn_low_z, &qfluxn_high_x, &qfluxn_high_y, &qfluxn_high_z, qfluxn.getPointer(),
             &qfluxs_low_x, &qfluxs_low_y, &qfluxs_low_z, &qfluxs_high_x, &qfluxs_high_y, &qfluxs_high_z, qfluxs.getPointer(),
             &qfluxt_low_x, &qfluxt_low_y, &qfluxt_low_z, &qfluxt_high_x, &qfluxt_high_y, &qfluxt_high_z, qfluxt.getPointer(),
             &qfluxb_low_x, &qfluxb_low_y, &qfluxb_low_z, &qfluxb_high_x, &qfluxb_high_y, &qfluxb_high_z, qfluxb.getPointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine rdomflux(idxlo, idxhi, l, oxi_low, oxi_high, oxi, 
     & omu_low, omu_high, omu, oeta_low, oeta_high, oeta, wt_low, 
     & wt_high, wt, cenint_low_x, cenint_low_y, cenint_low_z, 
     & cenint_high_x, cenint_high_y, cenint_high_z, cenint, plusX, 
     & plusY, plusZ, qfluxe_low_x, qfluxe_low_y, qfluxe_low_z, 
     & qfluxe_high_x, qfluxe_high_y, qfluxe_high_z, qfluxe, 
     & qfluxw_low_x, qfluxw_low_y, qfluxw_low_z, qfluxw_high_x, 
     & qfluxw_high_y, qfluxw_high_z, qfluxw, qfluxn_low_x, qfluxn_low_y
     & , qfluxn_low_z, qfluxn_high_x, qfluxn_high_y, qfluxn_high_z, 
     & qfluxn, qfluxs_low_x, qfluxs_low_y, qfluxs_low_z, qfluxs_high_x,
     &  qfluxs_high_y, qfluxs_high_z, qfluxs, qfluxt_low_x, 
     & qfluxt_low_y, qfluxt_low_z, qfluxt_high_x, qfluxt_high_y, 
     & qfluxt_high_z, qfluxt, qfluxb_low_x, qfluxb_low_y, qfluxb_low_z,
     &  qfluxb_high_x, qfluxb_high_y, qfluxb_high_z, qfluxb)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
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
      integer cenint_low_x, cenint_low_y, cenint_low_z, cenint_high_x, 
     & cenint_high_y, cenint_high_z
      double precision cenint(cenint_low_x:cenint_high_x, cenint_low_y:
     & cenint_high_y, cenint_low_z:cenint_high_z)
      logical*1 plusX
      logical*1 plusY
      logical*1 plusZ
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
#endif /* __cplusplus */

#endif /* fspec_rdomflux */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
