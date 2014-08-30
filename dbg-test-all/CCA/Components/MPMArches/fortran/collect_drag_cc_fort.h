
#ifndef fspec_collect_drag_cc
#define fspec_collect_drag_cc

#ifdef __cplusplus

extern "C" void collect_drag_cc_(int* su_dragx_cc_low_x, int* su_dragx_cc_low_y, int* su_dragx_cc_low_z, int* su_dragx_cc_high_x, int* su_dragx_cc_high_y, int* su_dragx_cc_high_z, double* su_dragx_cc_ptr,
                                 int* sp_dragx_cc_low_x, int* sp_dragx_cc_low_y, int* sp_dragx_cc_low_z, int* sp_dragx_cc_high_x, int* sp_dragx_cc_high_y, int* sp_dragx_cc_high_z, double* sp_dragx_cc_ptr,
                                 int* su_dragx_fc_low_x, int* su_dragx_fc_low_y, int* su_dragx_fc_low_z, int* su_dragx_fc_high_x, int* su_dragx_fc_high_y, int* su_dragx_fc_high_z, double* su_dragx_fc_ptr,
                                 int* sp_dragx_fc_low_x, int* sp_dragx_fc_low_y, int* sp_dragx_fc_low_z, int* sp_dragx_fc_high_x, int* sp_dragx_fc_high_y, int* sp_dragx_fc_high_z, double* sp_dragx_fc_ptr,
                                 int* ioff,
                                 int* joff,
                                 int* koff,
                                 int* valid_lo,
                                 int* valid_hi);

static void fort_collect_drag_cc( Uintah::CCVariable<double> & su_dragx_cc,
                                  Uintah::CCVariable<double> & sp_dragx_cc,
                                  const Uintah::Array3<double> & su_dragx_fc,
                                  const Uintah::Array3<double> & sp_dragx_fc,
                                  int & ioff,
                                  int & joff,
                                  int & koff,
                                  Uintah::IntVector & valid_lo,
                                  Uintah::IntVector & valid_hi )
{
  Uintah::IntVector su_dragx_cc_low = su_dragx_cc.getWindow()->getOffset();
  Uintah::IntVector su_dragx_cc_high = su_dragx_cc.getWindow()->getData()->size() + su_dragx_cc_low - Uintah::IntVector(1, 1, 1);
  int su_dragx_cc_low_x = su_dragx_cc_low.x();
  int su_dragx_cc_high_x = su_dragx_cc_high.x();
  int su_dragx_cc_low_y = su_dragx_cc_low.y();
  int su_dragx_cc_high_y = su_dragx_cc_high.y();
  int su_dragx_cc_low_z = su_dragx_cc_low.z();
  int su_dragx_cc_high_z = su_dragx_cc_high.z();
  Uintah::IntVector sp_dragx_cc_low = sp_dragx_cc.getWindow()->getOffset();
  Uintah::IntVector sp_dragx_cc_high = sp_dragx_cc.getWindow()->getData()->size() + sp_dragx_cc_low - Uintah::IntVector(1, 1, 1);
  int sp_dragx_cc_low_x = sp_dragx_cc_low.x();
  int sp_dragx_cc_high_x = sp_dragx_cc_high.x();
  int sp_dragx_cc_low_y = sp_dragx_cc_low.y();
  int sp_dragx_cc_high_y = sp_dragx_cc_high.y();
  int sp_dragx_cc_low_z = sp_dragx_cc_low.z();
  int sp_dragx_cc_high_z = sp_dragx_cc_high.z();
  Uintah::IntVector su_dragx_fc_low = su_dragx_fc.getWindow()->getOffset();
  Uintah::IntVector su_dragx_fc_high = su_dragx_fc.getWindow()->getData()->size() + su_dragx_fc_low - Uintah::IntVector(1, 1, 1);
  int su_dragx_fc_low_x = su_dragx_fc_low.x();
  int su_dragx_fc_high_x = su_dragx_fc_high.x();
  int su_dragx_fc_low_y = su_dragx_fc_low.y();
  int su_dragx_fc_high_y = su_dragx_fc_high.y();
  int su_dragx_fc_low_z = su_dragx_fc_low.z();
  int su_dragx_fc_high_z = su_dragx_fc_high.z();
  Uintah::IntVector sp_dragx_fc_low = sp_dragx_fc.getWindow()->getOffset();
  Uintah::IntVector sp_dragx_fc_high = sp_dragx_fc.getWindow()->getData()->size() + sp_dragx_fc_low - Uintah::IntVector(1, 1, 1);
  int sp_dragx_fc_low_x = sp_dragx_fc_low.x();
  int sp_dragx_fc_high_x = sp_dragx_fc_high.x();
  int sp_dragx_fc_low_y = sp_dragx_fc_low.y();
  int sp_dragx_fc_high_y = sp_dragx_fc_high.y();
  int sp_dragx_fc_low_z = sp_dragx_fc_low.z();
  int sp_dragx_fc_high_z = sp_dragx_fc_high.z();
  collect_drag_cc_( &su_dragx_cc_low_x, &su_dragx_cc_low_y, &su_dragx_cc_low_z, &su_dragx_cc_high_x, &su_dragx_cc_high_y, &su_dragx_cc_high_z, su_dragx_cc.getPointer(),
                    &sp_dragx_cc_low_x, &sp_dragx_cc_low_y, &sp_dragx_cc_low_z, &sp_dragx_cc_high_x, &sp_dragx_cc_high_y, &sp_dragx_cc_high_z, sp_dragx_cc.getPointer(),
                    &su_dragx_fc_low_x, &su_dragx_fc_low_y, &su_dragx_fc_low_z, &su_dragx_fc_high_x, &su_dragx_fc_high_y, &su_dragx_fc_high_z, const_cast<double*>(su_dragx_fc.getPointer()),
                    &sp_dragx_fc_low_x, &sp_dragx_fc_low_y, &sp_dragx_fc_low_z, &sp_dragx_fc_high_x, &sp_dragx_fc_high_y, &sp_dragx_fc_high_z, const_cast<double*>(sp_dragx_fc.getPointer()),
                    &ioff,
                    &joff,
                    &koff,
                    valid_lo.get_pointer(),
                    valid_hi.get_pointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine collect_drag_cc(su_dragx_cc_low_x, su_dragx_cc_low_y, 
     & su_dragx_cc_low_z, su_dragx_cc_high_x, su_dragx_cc_high_y, 
     & su_dragx_cc_high_z, su_dragx_cc, sp_dragx_cc_low_x, 
     & sp_dragx_cc_low_y, sp_dragx_cc_low_z, sp_dragx_cc_high_x, 
     & sp_dragx_cc_high_y, sp_dragx_cc_high_z, sp_dragx_cc, 
     & su_dragx_fc_low_x, su_dragx_fc_low_y, su_dragx_fc_low_z, 
     & su_dragx_fc_high_x, su_dragx_fc_high_y, su_dragx_fc_high_z, 
     & su_dragx_fc, sp_dragx_fc_low_x, sp_dragx_fc_low_y, 
     & sp_dragx_fc_low_z, sp_dragx_fc_high_x, sp_dragx_fc_high_y, 
     & sp_dragx_fc_high_z, sp_dragx_fc, ioff, joff, koff, valid_lo, 
     & valid_hi)

      implicit none
      integer su_dragx_cc_low_x, su_dragx_cc_low_y, su_dragx_cc_low_z, 
     & su_dragx_cc_high_x, su_dragx_cc_high_y, su_dragx_cc_high_z
      double precision su_dragx_cc(su_dragx_cc_low_x:su_dragx_cc_high_x
     & , su_dragx_cc_low_y:su_dragx_cc_high_y, su_dragx_cc_low_z:
     & su_dragx_cc_high_z)
      integer sp_dragx_cc_low_x, sp_dragx_cc_low_y, sp_dragx_cc_low_z, 
     & sp_dragx_cc_high_x, sp_dragx_cc_high_y, sp_dragx_cc_high_z
      double precision sp_dragx_cc(sp_dragx_cc_low_x:sp_dragx_cc_high_x
     & , sp_dragx_cc_low_y:sp_dragx_cc_high_y, sp_dragx_cc_low_z:
     & sp_dragx_cc_high_z)
      integer su_dragx_fc_low_x, su_dragx_fc_low_y, su_dragx_fc_low_z, 
     & su_dragx_fc_high_x, su_dragx_fc_high_y, su_dragx_fc_high_z
      double precision su_dragx_fc(su_dragx_fc_low_x:su_dragx_fc_high_x
     & , su_dragx_fc_low_y:su_dragx_fc_high_y, su_dragx_fc_low_z:
     & su_dragx_fc_high_z)
      integer sp_dragx_fc_low_x, sp_dragx_fc_low_y, sp_dragx_fc_low_z, 
     & sp_dragx_fc_high_x, sp_dragx_fc_high_y, sp_dragx_fc_high_z
      double precision sp_dragx_fc(sp_dragx_fc_low_x:sp_dragx_fc_high_x
     & , sp_dragx_fc_low_y:sp_dragx_fc_high_y, sp_dragx_fc_low_z:
     & sp_dragx_fc_high_z)
      integer ioff
      integer joff
      integer koff
      integer valid_lo(3)
      integer valid_hi(3)
#endif /* __cplusplus */

#endif /* fspec_collect_drag_cc */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
