
#ifndef fspec_wallbc
#define fspec_wallbc

#ifdef __cplusplus

#include <CCA/Components/Arches/fortran/FortranNameMangle.h>

extern "C" void F_wallbc(int* idxLo,
                       int* idxHi,
                       int* ae_low_x, int* ae_low_y, int* ae_low_z, int* ae_high_x, int* ae_high_y, int* ae_high_z, double* ae_ptr,
                       int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
                       int* an_low_x, int* an_low_y, int* an_low_z, int* an_high_x, int* an_high_y, int* an_high_z, double* an_ptr,
                       int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
                       int* at_low_x, int* at_low_y, int* at_low_z, int* at_high_x, int* at_high_y, int* at_high_z, double* at_ptr,
                       int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
                       int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                       int* wall,
                       bool* xminus,
                       bool* xplus,
                       bool* yminus,
                       bool* yplus,
                       bool* zminus,
                       bool* zplus);

static void fort_wallbc( Uintah::IntVector & idxLo,
                         Uintah::IntVector & idxHi,
                         Uintah::CCVariable<double> & ae,
                         Uintah::CCVariable<double> & aw,
                         Uintah::CCVariable<double> & an,
                         Uintah::CCVariable<double> & as,
                         Uintah::CCVariable<double> & at,
                         Uintah::CCVariable<double> & ab,
                         Uintah::CCVariable<int> & pcell,
                         int & wall,
                         bool & xminus,
                         bool & xplus,
                         bool & yminus,
                         bool & yplus,
                         bool & zminus,
                         bool & zplus )
{
  Uintah::IntVector ae_low = ae.getWindow()->getOffset();
  Uintah::IntVector ae_high = ae.getWindow()->getData()->size() + ae_low - Uintah::IntVector(1, 1, 1);
  int ae_low_x = ae_low.x();
  int ae_high_x = ae_high.x();
  int ae_low_y = ae_low.y();
  int ae_high_y = ae_high.y();
  int ae_low_z = ae_low.z();
  int ae_high_z = ae_high.z();
  Uintah::IntVector aw_low = aw.getWindow()->getOffset();
  Uintah::IntVector aw_high = aw.getWindow()->getData()->size() + aw_low - Uintah::IntVector(1, 1, 1);
  int aw_low_x = aw_low.x();
  int aw_high_x = aw_high.x();
  int aw_low_y = aw_low.y();
  int aw_high_y = aw_high.y();
  int aw_low_z = aw_low.z();
  int aw_high_z = aw_high.z();
  Uintah::IntVector an_low = an.getWindow()->getOffset();
  Uintah::IntVector an_high = an.getWindow()->getData()->size() + an_low - Uintah::IntVector(1, 1, 1);
  int an_low_x = an_low.x();
  int an_high_x = an_high.x();
  int an_low_y = an_low.y();
  int an_high_y = an_high.y();
  int an_low_z = an_low.z();
  int an_high_z = an_high.z();
  Uintah::IntVector as_low = as.getWindow()->getOffset();
  Uintah::IntVector as_high = as.getWindow()->getData()->size() + as_low - Uintah::IntVector(1, 1, 1);
  int as_low_x = as_low.x();
  int as_high_x = as_high.x();
  int as_low_y = as_low.y();
  int as_high_y = as_high.y();
  int as_low_z = as_low.z();
  int as_high_z = as_high.z();
  Uintah::IntVector at_low = at.getWindow()->getOffset();
  Uintah::IntVector at_high = at.getWindow()->getData()->size() + at_low - Uintah::IntVector(1, 1, 1);
  int at_low_x = at_low.x();
  int at_high_x = at_high.x();
  int at_low_y = at_low.y();
  int at_high_y = at_high.y();
  int at_low_z = at_low.z();
  int at_high_z = at_high.z();
  Uintah::IntVector ab_low = ab.getWindow()->getOffset();
  Uintah::IntVector ab_high = ab.getWindow()->getData()->size() + ab_low - Uintah::IntVector(1, 1, 1);
  int ab_low_x = ab_low.x();
  int ab_high_x = ab_high.x();
  int ab_low_y = ab_low.y();
  int ab_high_y = ab_high.y();
  int ab_low_z = ab_low.z();
  int ab_high_z = ab_high.z();
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  F_wallbc( idxLo.get_pointer(),
          idxHi.get_pointer(),
          &ae_low_x, &ae_low_y, &ae_low_z, &ae_high_x, &ae_high_y, &ae_high_z, ae.getPointer(),
          &aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
          &an_low_x, &an_low_y, &an_low_z, &an_high_x, &an_high_y, &an_high_z, an.getPointer(),
          &as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
          &at_low_x, &at_low_y, &at_low_z, &at_high_x, &at_high_y, &at_high_z, at.getPointer(),
          &ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
          &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, pcell.getPointer(),
          &wall,
          &xminus,
          &xplus,
          &yminus,
          &yplus,
          &zminus,
          &zplus );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine WALLBC(idxLo, idxHi, ae_low_x, ae_low_y, ae_low_z,
     & ae_high_x, ae_high_y, ae_high_z, ae, aw_low_x, aw_low_y, 
     & aw_low_z, aw_high_x, aw_high_y, aw_high_z, aw, an_low_x, 
     & an_low_y, an_low_z, an_high_x, an_high_y, an_high_z, an, 
     & as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, as_high_z, 
     & as, at_low_x, at_low_y, at_low_z, at_high_x, at_high_y, 
     & at_high_z, at, ab_low_x, ab_low_y, ab_low_z, ab_high_x, 
     & ab_high_y, ab_high_z, ab, pcell_low_x, pcell_low_y, pcell_low_z,
     &  pcell_high_x, pcell_high_y, pcell_high_z, pcell, wall, xminus, 
     & xplus, yminus, yplus, zminus, zplus)

      implicit none
      integer idxLo(3)
      integer idxHi(3)
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
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer wall
      logical*1 xminus
      logical*1 xplus
      logical*1 yminus
      logical*1 yplus
      logical*1 zminus
      logical*1 zplus
#endif /* __cplusplus */

#endif /* fspec_wallbc */
