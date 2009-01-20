
#ifndef fspec_rdombc
#define fspec_rdombc

#ifdef __cplusplus

extern "C" void rdombc_(int* idxlo,
int* idxhi,
int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
int* ffield,
int* tg_low_x, int* tg_low_y, int* tg_low_z, int* tg_high_x, int* tg_high_y, int* tg_high_z, double* tg_ptr,
int* abskg_low_x, int* abskg_low_y, int* abskg_low_z, int* abskg_high_x, int* abskg_high_y, int* abskg_high_z, double* abskg_ptr,
bool* xminus,
bool* xplus,
bool* yminus,
bool* yplus,
bool* zminus,
bool* zplus,
bool* lprobone,
bool* lprobtwo,
bool* lprobthree);

static void fort_rdombc(IntVector& idxlo,
IntVector& idxhi,
constCCVariable<int>& pcell,
int& ffield,
CCVariable<double>& tg,
CCVariable<double>& abskg,
bool& xminus,
bool& xplus,
bool& yminus,
bool& yplus,
bool& zminus,
bool& zplus,
bool& lprobone,
bool& lprobtwo,
bool& lprobthree)
{
  IntVector pcell_low = pcell.getWindow()->getOffset();
  IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
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
  rdombc_(idxlo.get_pointer(),
idxhi.get_pointer(),
&pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
&ffield,
&tg_low_x, &tg_low_y, &tg_low_z, &tg_high_x, &tg_high_y, &tg_high_z, tg.getPointer(),
&abskg_low_x, &abskg_low_y, &abskg_low_z, &abskg_high_x, &abskg_high_y, &abskg_high_z, abskg.getPointer(),
&xminus,
&xplus,
&yminus,
&yplus,
&zminus,
&zplus,
&lprobone,
&lprobtwo,
&lprobthree);
}

#else /* !__cplusplus */
C Assuming this is fortran code

      subroutine rdombc(idxlo, idxhi, pcell_low_x, pcell_low_y, 
     & pcell_low_z, pcell_high_x, pcell_high_y, pcell_high_z, pcell, 
     & ffield, tg_low_x, tg_low_y, tg_low_z, tg_high_x, tg_high_y, 
     & tg_high_z, tg, abskg_low_x, abskg_low_y, abskg_low_z, 
     & abskg_high_x, abskg_high_y, abskg_high_z, abskg, xminus, xplus, 
     & yminus, yplus, zminus, zplus, lprobone, lprobtwo, lprobthree)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer ffield
      integer tg_low_x, tg_low_y, tg_low_z, tg_high_x, tg_high_y, 
     & tg_high_z
      double precision tg(tg_low_x:tg_high_x, tg_low_y:tg_high_y, 
     & tg_low_z:tg_high_z)
      integer abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z
      double precision abskg(abskg_low_x:abskg_high_x, abskg_low_y:
     & abskg_high_y, abskg_low_z:abskg_high_z)
      logical*1 xminus
      logical*1 xplus
      logical*1 yminus
      logical*1 yplus
      logical*1 zminus
      logical*1 zplus
      logical*1 lprobone
      logical*1 lprobtwo
      logical*1 lprobthree
#endif /* __cplusplus */

#endif /* fspec_rdombc */

#ifndef PASS1
#define PASS1(x) x/**/_low, x/**/_high, x
#endif
#ifndef PASS3
#define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
