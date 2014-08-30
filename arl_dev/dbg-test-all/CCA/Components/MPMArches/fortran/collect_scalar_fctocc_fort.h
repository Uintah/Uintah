
#ifndef fspec_collect_scalar_fctocc
#define fspec_collect_scalar_fctocc

#ifdef __cplusplus

extern "C" void collect_scalar_fctocc_(int* su_enth_cc_low_x, int* su_enth_cc_low_y, int* su_enth_cc_low_z, int* su_enth_cc_high_x, int* su_enth_cc_high_y, int* su_enth_cc_high_z, double* su_enth_cc_ptr,
                                       int* sp_enth_cc_low_x, int* sp_enth_cc_low_y, int* sp_enth_cc_low_z, int* sp_enth_cc_high_x, int* sp_enth_cc_high_y, int* sp_enth_cc_high_z, double* sp_enth_cc_ptr,
                                       int* su_enth_fcx_low_x, int* su_enth_fcx_low_y, int* su_enth_fcx_low_z, int* su_enth_fcx_high_x, int* su_enth_fcx_high_y, int* su_enth_fcx_high_z, double* su_enth_fcx_ptr,
                                       int* sp_enth_fcx_low_x, int* sp_enth_fcx_low_y, int* sp_enth_fcx_low_z, int* sp_enth_fcx_high_x, int* sp_enth_fcx_high_y, int* sp_enth_fcx_high_z, double* sp_enth_fcx_ptr,
                                       int* su_enth_fcy_low_x, int* su_enth_fcy_low_y, int* su_enth_fcy_low_z, int* su_enth_fcy_high_x, int* su_enth_fcy_high_y, int* su_enth_fcy_high_z, double* su_enth_fcy_ptr,
                                       int* sp_enth_fcy_low_x, int* sp_enth_fcy_low_y, int* sp_enth_fcy_low_z, int* sp_enth_fcy_high_x, int* sp_enth_fcy_high_y, int* sp_enth_fcy_high_z, double* sp_enth_fcy_ptr,
                                       int* su_enth_fcz_low_x, int* su_enth_fcz_low_y, int* su_enth_fcz_low_z, int* su_enth_fcz_high_x, int* su_enth_fcz_high_y, int* su_enth_fcz_high_z, double* su_enth_fcz_ptr,
                                       int* sp_enth_fcz_low_x, int* sp_enth_fcz_low_y, int* sp_enth_fcz_low_z, int* sp_enth_fcz_high_x, int* sp_enth_fcz_high_y, int* sp_enth_fcz_high_z, double* sp_enth_fcz_ptr,
                                       int* valid_lo,
                                       int* valid_hi);

static void fort_collect_scalar_fctocc( Uintah::CCVariable<double> & su_enth_cc,
                                        Uintah::CCVariable<double> & sp_enth_cc,
                                        Uintah::constSFCXVariable<double> & su_enth_fcx,
                                        Uintah::constSFCXVariable<double> & sp_enth_fcx,
                                        Uintah::constSFCYVariable<double> & su_enth_fcy,
                                        Uintah::constSFCYVariable<double> & sp_enth_fcy,
                                        Uintah::constSFCZVariable<double> & su_enth_fcz,
                                        Uintah::constSFCZVariable<double> & sp_enth_fcz,
                                        Uintah::IntVector & valid_lo,
                                        Uintah::IntVector & valid_hi )
{
  Uintah::IntVector su_enth_cc_low = su_enth_cc.getWindow()->getOffset();
  Uintah::IntVector su_enth_cc_high = su_enth_cc.getWindow()->getData()->size() + su_enth_cc_low - Uintah::IntVector(1, 1, 1);
  int su_enth_cc_low_x = su_enth_cc_low.x();
  int su_enth_cc_high_x = su_enth_cc_high.x();
  int su_enth_cc_low_y = su_enth_cc_low.y();
  int su_enth_cc_high_y = su_enth_cc_high.y();
  int su_enth_cc_low_z = su_enth_cc_low.z();
  int su_enth_cc_high_z = su_enth_cc_high.z();
  Uintah::IntVector sp_enth_cc_low = sp_enth_cc.getWindow()->getOffset();
  Uintah::IntVector sp_enth_cc_high = sp_enth_cc.getWindow()->getData()->size() + sp_enth_cc_low - Uintah::IntVector(1, 1, 1);
  int sp_enth_cc_low_x = sp_enth_cc_low.x();
  int sp_enth_cc_high_x = sp_enth_cc_high.x();
  int sp_enth_cc_low_y = sp_enth_cc_low.y();
  int sp_enth_cc_high_y = sp_enth_cc_high.y();
  int sp_enth_cc_low_z = sp_enth_cc_low.z();
  int sp_enth_cc_high_z = sp_enth_cc_high.z();
  Uintah::IntVector su_enth_fcx_low = su_enth_fcx.getWindow()->getOffset();
  Uintah::IntVector su_enth_fcx_high = su_enth_fcx.getWindow()->getData()->size() + su_enth_fcx_low - Uintah::IntVector(1, 1, 1);
  int su_enth_fcx_low_x = su_enth_fcx_low.x();
  int su_enth_fcx_high_x = su_enth_fcx_high.x();
  int su_enth_fcx_low_y = su_enth_fcx_low.y();
  int su_enth_fcx_high_y = su_enth_fcx_high.y();
  int su_enth_fcx_low_z = su_enth_fcx_low.z();
  int su_enth_fcx_high_z = su_enth_fcx_high.z();
  Uintah::IntVector sp_enth_fcx_low = sp_enth_fcx.getWindow()->getOffset();
  Uintah::IntVector sp_enth_fcx_high = sp_enth_fcx.getWindow()->getData()->size() + sp_enth_fcx_low - Uintah::IntVector(1, 1, 1);
  int sp_enth_fcx_low_x = sp_enth_fcx_low.x();
  int sp_enth_fcx_high_x = sp_enth_fcx_high.x();
  int sp_enth_fcx_low_y = sp_enth_fcx_low.y();
  int sp_enth_fcx_high_y = sp_enth_fcx_high.y();
  int sp_enth_fcx_low_z = sp_enth_fcx_low.z();
  int sp_enth_fcx_high_z = sp_enth_fcx_high.z();
  Uintah::IntVector su_enth_fcy_low = su_enth_fcy.getWindow()->getOffset();
  Uintah::IntVector su_enth_fcy_high = su_enth_fcy.getWindow()->getData()->size() + su_enth_fcy_low - Uintah::IntVector(1, 1, 1);
  int su_enth_fcy_low_x = su_enth_fcy_low.x();
  int su_enth_fcy_high_x = su_enth_fcy_high.x();
  int su_enth_fcy_low_y = su_enth_fcy_low.y();
  int su_enth_fcy_high_y = su_enth_fcy_high.y();
  int su_enth_fcy_low_z = su_enth_fcy_low.z();
  int su_enth_fcy_high_z = su_enth_fcy_high.z();
  Uintah::IntVector sp_enth_fcy_low = sp_enth_fcy.getWindow()->getOffset();
  Uintah::IntVector sp_enth_fcy_high = sp_enth_fcy.getWindow()->getData()->size() + sp_enth_fcy_low - Uintah::IntVector(1, 1, 1);
  int sp_enth_fcy_low_x = sp_enth_fcy_low.x();
  int sp_enth_fcy_high_x = sp_enth_fcy_high.x();
  int sp_enth_fcy_low_y = sp_enth_fcy_low.y();
  int sp_enth_fcy_high_y = sp_enth_fcy_high.y();
  int sp_enth_fcy_low_z = sp_enth_fcy_low.z();
  int sp_enth_fcy_high_z = sp_enth_fcy_high.z();
  Uintah::IntVector su_enth_fcz_low = su_enth_fcz.getWindow()->getOffset();
  Uintah::IntVector su_enth_fcz_high = su_enth_fcz.getWindow()->getData()->size() + su_enth_fcz_low - Uintah::IntVector(1, 1, 1);
  int su_enth_fcz_low_x = su_enth_fcz_low.x();
  int su_enth_fcz_high_x = su_enth_fcz_high.x();
  int su_enth_fcz_low_y = su_enth_fcz_low.y();
  int su_enth_fcz_high_y = su_enth_fcz_high.y();
  int su_enth_fcz_low_z = su_enth_fcz_low.z();
  int su_enth_fcz_high_z = su_enth_fcz_high.z();
  Uintah::IntVector sp_enth_fcz_low = sp_enth_fcz.getWindow()->getOffset();
  Uintah::IntVector sp_enth_fcz_high = sp_enth_fcz.getWindow()->getData()->size() + sp_enth_fcz_low - Uintah::IntVector(1, 1, 1);
  int sp_enth_fcz_low_x = sp_enth_fcz_low.x();
  int sp_enth_fcz_high_x = sp_enth_fcz_high.x();
  int sp_enth_fcz_low_y = sp_enth_fcz_low.y();
  int sp_enth_fcz_high_y = sp_enth_fcz_high.y();
  int sp_enth_fcz_low_z = sp_enth_fcz_low.z();
  int sp_enth_fcz_high_z = sp_enth_fcz_high.z();
  collect_scalar_fctocc_( &su_enth_cc_low_x, &su_enth_cc_low_y, &su_enth_cc_low_z, &su_enth_cc_high_x, &su_enth_cc_high_y, &su_enth_cc_high_z, su_enth_cc.getPointer(),
                          &sp_enth_cc_low_x, &sp_enth_cc_low_y, &sp_enth_cc_low_z, &sp_enth_cc_high_x, &sp_enth_cc_high_y, &sp_enth_cc_high_z, sp_enth_cc.getPointer(),
                          &su_enth_fcx_low_x, &su_enth_fcx_low_y, &su_enth_fcx_low_z, &su_enth_fcx_high_x, &su_enth_fcx_high_y, &su_enth_fcx_high_z, const_cast<double*>(su_enth_fcx.getPointer()),
                          &sp_enth_fcx_low_x, &sp_enth_fcx_low_y, &sp_enth_fcx_low_z, &sp_enth_fcx_high_x, &sp_enth_fcx_high_y, &sp_enth_fcx_high_z, const_cast<double*>(sp_enth_fcx.getPointer()),
                          &su_enth_fcy_low_x, &su_enth_fcy_low_y, &su_enth_fcy_low_z, &su_enth_fcy_high_x, &su_enth_fcy_high_y, &su_enth_fcy_high_z, const_cast<double*>(su_enth_fcy.getPointer()),
                          &sp_enth_fcy_low_x, &sp_enth_fcy_low_y, &sp_enth_fcy_low_z, &sp_enth_fcy_high_x, &sp_enth_fcy_high_y, &sp_enth_fcy_high_z, const_cast<double*>(sp_enth_fcy.getPointer()),
                          &su_enth_fcz_low_x, &su_enth_fcz_low_y, &su_enth_fcz_low_z, &su_enth_fcz_high_x, &su_enth_fcz_high_y, &su_enth_fcz_high_z, const_cast<double*>(su_enth_fcz.getPointer()),
                          &sp_enth_fcz_low_x, &sp_enth_fcz_low_y, &sp_enth_fcz_low_z, &sp_enth_fcz_high_x, &sp_enth_fcz_high_y, &sp_enth_fcz_high_z, const_cast<double*>(sp_enth_fcz.getPointer()),
                          valid_lo.get_pointer(),
                          valid_hi.get_pointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine collect_scalar_fctocc(su_enth_cc_low_x, 
     & su_enth_cc_low_y, su_enth_cc_low_z, su_enth_cc_high_x, 
     & su_enth_cc_high_y, su_enth_cc_high_z, su_enth_cc, 
     & sp_enth_cc_low_x, sp_enth_cc_low_y, sp_enth_cc_low_z, 
     & sp_enth_cc_high_x, sp_enth_cc_high_y, sp_enth_cc_high_z, 
     & sp_enth_cc, su_enth_fcx_low_x, su_enth_fcx_low_y, 
     & su_enth_fcx_low_z, su_enth_fcx_high_x, su_enth_fcx_high_y, 
     & su_enth_fcx_high_z, su_enth_fcx, sp_enth_fcx_low_x, 
     & sp_enth_fcx_low_y, sp_enth_fcx_low_z, sp_enth_fcx_high_x, 
     & sp_enth_fcx_high_y, sp_enth_fcx_high_z, sp_enth_fcx, 
     & su_enth_fcy_low_x, su_enth_fcy_low_y, su_enth_fcy_low_z, 
     & su_enth_fcy_high_x, su_enth_fcy_high_y, su_enth_fcy_high_z, 
     & su_enth_fcy, sp_enth_fcy_low_x, sp_enth_fcy_low_y, 
     & sp_enth_fcy_low_z, sp_enth_fcy_high_x, sp_enth_fcy_high_y, 
     & sp_enth_fcy_high_z, sp_enth_fcy, su_enth_fcz_low_x, 
     & su_enth_fcz_low_y, su_enth_fcz_low_z, su_enth_fcz_high_x, 
     & su_enth_fcz_high_y, su_enth_fcz_high_z, su_enth_fcz, 
     & sp_enth_fcz_low_x, sp_enth_fcz_low_y, sp_enth_fcz_low_z, 
     & sp_enth_fcz_high_x, sp_enth_fcz_high_y, sp_enth_fcz_high_z, 
     & sp_enth_fcz, valid_lo, valid_hi)

      implicit none
      integer su_enth_cc_low_x, su_enth_cc_low_y, su_enth_cc_low_z, 
     & su_enth_cc_high_x, su_enth_cc_high_y, su_enth_cc_high_z
      double precision su_enth_cc(su_enth_cc_low_x:su_enth_cc_high_x, 
     & su_enth_cc_low_y:su_enth_cc_high_y, su_enth_cc_low_z:
     & su_enth_cc_high_z)
      integer sp_enth_cc_low_x, sp_enth_cc_low_y, sp_enth_cc_low_z, 
     & sp_enth_cc_high_x, sp_enth_cc_high_y, sp_enth_cc_high_z
      double precision sp_enth_cc(sp_enth_cc_low_x:sp_enth_cc_high_x, 
     & sp_enth_cc_low_y:sp_enth_cc_high_y, sp_enth_cc_low_z:
     & sp_enth_cc_high_z)
      integer su_enth_fcx_low_x, su_enth_fcx_low_y, su_enth_fcx_low_z, 
     & su_enth_fcx_high_x, su_enth_fcx_high_y, su_enth_fcx_high_z
      double precision su_enth_fcx(su_enth_fcx_low_x:su_enth_fcx_high_x
     & , su_enth_fcx_low_y:su_enth_fcx_high_y, su_enth_fcx_low_z:
     & su_enth_fcx_high_z)
      integer sp_enth_fcx_low_x, sp_enth_fcx_low_y, sp_enth_fcx_low_z, 
     & sp_enth_fcx_high_x, sp_enth_fcx_high_y, sp_enth_fcx_high_z
      double precision sp_enth_fcx(sp_enth_fcx_low_x:sp_enth_fcx_high_x
     & , sp_enth_fcx_low_y:sp_enth_fcx_high_y, sp_enth_fcx_low_z:
     & sp_enth_fcx_high_z)
      integer su_enth_fcy_low_x, su_enth_fcy_low_y, su_enth_fcy_low_z, 
     & su_enth_fcy_high_x, su_enth_fcy_high_y, su_enth_fcy_high_z
      double precision su_enth_fcy(su_enth_fcy_low_x:su_enth_fcy_high_x
     & , su_enth_fcy_low_y:su_enth_fcy_high_y, su_enth_fcy_low_z:
     & su_enth_fcy_high_z)
      integer sp_enth_fcy_low_x, sp_enth_fcy_low_y, sp_enth_fcy_low_z, 
     & sp_enth_fcy_high_x, sp_enth_fcy_high_y, sp_enth_fcy_high_z
      double precision sp_enth_fcy(sp_enth_fcy_low_x:sp_enth_fcy_high_x
     & , sp_enth_fcy_low_y:sp_enth_fcy_high_y, sp_enth_fcy_low_z:
     & sp_enth_fcy_high_z)
      integer su_enth_fcz_low_x, su_enth_fcz_low_y, su_enth_fcz_low_z, 
     & su_enth_fcz_high_x, su_enth_fcz_high_y, su_enth_fcz_high_z
      double precision su_enth_fcz(su_enth_fcz_low_x:su_enth_fcz_high_x
     & , su_enth_fcz_low_y:su_enth_fcz_high_y, su_enth_fcz_low_z:
     & su_enth_fcz_high_z)
      integer sp_enth_fcz_low_x, sp_enth_fcz_low_y, sp_enth_fcz_low_z, 
     & sp_enth_fcz_high_x, sp_enth_fcz_high_y, sp_enth_fcz_high_z
      double precision sp_enth_fcz(sp_enth_fcz_low_x:sp_enth_fcz_high_x
     & , sp_enth_fcz_low_y:sp_enth_fcz_high_y, sp_enth_fcz_low_z:
     & sp_enth_fcz_high_z)
      integer valid_lo(3)
      integer valid_hi(3)
#endif /* __cplusplus */

#endif /* fspec_collect_scalar_fctocc */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
