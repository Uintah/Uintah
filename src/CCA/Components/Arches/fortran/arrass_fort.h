
#ifndef fspec_arrass
#define fspec_arrass

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_arrass(int* aa_low_x, int* aa_low_y, int* aa_low_z, int* aa_high_x, int* aa_high_y, int* aa_high_z, double* aa_ptr,
                       double* alpha,
                       int* valid_lo,
                       int* valid_hi);

static void fort_arrass( Uintah::Array3<double> & aa,
                         double & alpha,
                         Uintah::IntVector & valid_lo,
                         Uintah::IntVector & valid_hi )
{
  Uintah::IntVector aa_low = aa.getWindow()->getOffset();
  Uintah::IntVector aa_high = aa.getWindow()->getData()->size() + aa_low - Uintah::IntVector(1, 1, 1);
  int aa_low_x = aa_low.x();
  int aa_high_x = aa_high.x();
  int aa_low_y = aa_low.y();
  int aa_high_y = aa_high.y();
  int aa_low_z = aa_low.z();
  int aa_high_z = aa_high.z();
  F_arrass( &aa_low_x, &aa_low_y, &aa_low_z, &aa_high_x, &aa_high_y, &aa_high_z, aa.getPointer(),
          &alpha,
          valid_lo.get_pointer(),
          valid_hi.get_pointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine ARRASS(aa_low_x, aa_low_y, aa_low_z, aa_high_x,
     & aa_high_y, aa_high_z, aa, alpha, valid_lo, valid_hi)

      implicit none
      integer aa_low_x, aa_low_y, aa_low_z, aa_high_x, aa_high_y, 
     & aa_high_z
      double precision aa(aa_low_x:aa_high_x, aa_low_y:aa_high_y, 
     & aa_low_z:aa_high_z)
      double precision alpha
      integer valid_lo(3)
      integer valid_hi(3)
#endif /* __cplusplus */

#endif /* fspec_arrass */
