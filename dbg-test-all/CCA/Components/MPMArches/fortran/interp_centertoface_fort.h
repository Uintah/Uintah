
#ifndef fspec_interp_centertoface
#define fspec_interp_centertoface

#ifdef __cplusplus

extern "C" void interp_centertoface_(int* phi_fc_low_x, int* phi_fc_low_y, int* phi_fc_low_z, int* phi_fc_high_x, int* phi_fc_high_y, int* phi_fc_high_z, double* phi_fc_ptr,
                                     int* phi_cc_low_x, int* phi_cc_low_y, int* phi_cc_low_z, int* phi_cc_high_x, int* phi_cc_high_y, int* phi_cc_high_z, double* phi_cc_ptr,
                                     int* ioff,
                                     int* joff,
                                     int* koff,
                                     int* valid_lo,
                                     int* valid_hi);

static void fort_interp_centertoface( Uintah::Array3<double> & phi_fc,
                                      Uintah::constCCVariable<double> & phi_cc,
                                      int & ioff,
                                      int & joff,
                                      int & koff,
                                      Uintah::IntVector & valid_lo,
                                      Uintah::IntVector & valid_hi )
{
  Uintah::IntVector phi_fc_low = phi_fc.getWindow()->getOffset();
  Uintah::IntVector phi_fc_high = phi_fc.getWindow()->getData()->size() + phi_fc_low - Uintah::IntVector(1, 1, 1);
  int phi_fc_low_x = phi_fc_low.x();
  int phi_fc_high_x = phi_fc_high.x();
  int phi_fc_low_y = phi_fc_low.y();
  int phi_fc_high_y = phi_fc_high.y();
  int phi_fc_low_z = phi_fc_low.z();
  int phi_fc_high_z = phi_fc_high.z();
  Uintah::IntVector phi_cc_low = phi_cc.getWindow()->getOffset();
  Uintah::IntVector phi_cc_high = phi_cc.getWindow()->getData()->size() + phi_cc_low - Uintah::IntVector(1, 1, 1);
  int phi_cc_low_x = phi_cc_low.x();
  int phi_cc_high_x = phi_cc_high.x();
  int phi_cc_low_y = phi_cc_low.y();
  int phi_cc_high_y = phi_cc_high.y();
  int phi_cc_low_z = phi_cc_low.z();
  int phi_cc_high_z = phi_cc_high.z();
  interp_centertoface_( &phi_fc_low_x, &phi_fc_low_y, &phi_fc_low_z, &phi_fc_high_x, &phi_fc_high_y, &phi_fc_high_z, phi_fc.getPointer(),
                        &phi_cc_low_x, &phi_cc_low_y, &phi_cc_low_z, &phi_cc_high_x, &phi_cc_high_y, &phi_cc_high_z, const_cast<double*>(phi_cc.getPointer()),
                        &ioff,
                        &joff,
                        &koff,
                        valid_lo.get_pointer(),
                        valid_hi.get_pointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine interp_centertoface(phi_fc_low_x, phi_fc_low_y, 
     & phi_fc_low_z, phi_fc_high_x, phi_fc_high_y, phi_fc_high_z, 
     & phi_fc, phi_cc_low_x, phi_cc_low_y, phi_cc_low_z, phi_cc_high_x,
     &  phi_cc_high_y, phi_cc_high_z, phi_cc, ioff, joff, koff, 
     & valid_lo, valid_hi)

      implicit none
      integer phi_fc_low_x, phi_fc_low_y, phi_fc_low_z, phi_fc_high_x, 
     & phi_fc_high_y, phi_fc_high_z
      double precision phi_fc(phi_fc_low_x:phi_fc_high_x, phi_fc_low_y:
     & phi_fc_high_y, phi_fc_low_z:phi_fc_high_z)
      integer phi_cc_low_x, phi_cc_low_y, phi_cc_low_z, phi_cc_high_x, 
     & phi_cc_high_y, phi_cc_high_z
      double precision phi_cc(phi_cc_low_x:phi_cc_high_x, phi_cc_low_y:
     & phi_cc_high_y, phi_cc_low_z:phi_cc_high_z)
      integer ioff
      integer joff
      integer koff
      integer valid_lo(3)
      integer valid_hi(3)
#endif /* __cplusplus */

#endif /* fspec_interp_centertoface */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
