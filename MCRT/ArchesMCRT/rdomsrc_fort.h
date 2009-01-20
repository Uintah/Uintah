
#ifndef fspec_rdomsrc
#define fspec_rdomsrc

#ifdef __cplusplus

extern "C" void rdomsrc_(int* idxlo,
int* idxhi,
int* abskg_low_x, int* abskg_low_y, int* abskg_low_z, int* abskg_high_x, int* abskg_high_y, int* abskg_high_z, double* abskg_ptr,
int* esrcg_low_x, int* esrcg_low_y, int* esrcg_low_z, int* esrcg_high_x, int* esrcg_high_y, int* esrcg_high_z, double* esrcg_ptr,
int* volq_low_x, int* volq_low_y, int* volq_low_z, int* volq_high_x, int* volq_high_y, int* volq_high_z, double* volq_ptr,
int* src_low_x, int* src_low_y, int* src_low_z, int* src_high_x, int* src_high_y, int* src_high_z, double* src_ptr);

static void fort_rdomsrc(IntVector& idxlo,
IntVector& idxhi,
CCVariable<double>& abskg,
CCVariable<double>& esrcg,
CCVariable<double>& volq,
CCVariable<double>& src)
{
  IntVector abskg_low = abskg.getWindow()->getOffset();
  IntVector abskg_high = abskg.getWindow()->getData()->size() + abskg_low - IntVector(1, 1, 1);
  int abskg_low_x = abskg_low.x();
  int abskg_high_x = abskg_high.x();
  int abskg_low_y = abskg_low.y();
  int abskg_high_y = abskg_high.y();
  int abskg_low_z = abskg_low.z();
  int abskg_high_z = abskg_high.z();
  IntVector esrcg_low = esrcg.getWindow()->getOffset();
  IntVector esrcg_high = esrcg.getWindow()->getData()->size() + esrcg_low - IntVector(1, 1, 1);
  int esrcg_low_x = esrcg_low.x();
  int esrcg_high_x = esrcg_high.x();
  int esrcg_low_y = esrcg_low.y();
  int esrcg_high_y = esrcg_high.y();
  int esrcg_low_z = esrcg_low.z();
  int esrcg_high_z = esrcg_high.z();
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
  rdomsrc_(idxlo.get_pointer(),
idxhi.get_pointer(),
&abskg_low_x, &abskg_low_y, &abskg_low_z, &abskg_high_x, &abskg_high_y, &abskg_high_z, abskg.getPointer(),
&esrcg_low_x, &esrcg_low_y, &esrcg_low_z, &esrcg_high_x, &esrcg_high_y, &esrcg_high_z, esrcg.getPointer(),
&volq_low_x, &volq_low_y, &volq_low_z, &volq_high_x, &volq_high_y, &volq_high_z, volq.getPointer(),
&src_low_x, &src_low_y, &src_low_z, &src_high_x, &src_high_y, &src_high_z, src.getPointer());
}

#else /* !__cplusplus */
C Assuming this is fortran code

      subroutine rdomsrc(idxlo, idxhi, abskg_low_x, abskg_low_y, 
     & abskg_low_z, abskg_high_x, abskg_high_y, abskg_high_z, abskg, 
     & esrcg_low_x, esrcg_low_y, esrcg_low_z, esrcg_high_x, 
     & esrcg_high_y, esrcg_high_z, esrcg, volq_low_x, volq_low_y, 
     & volq_low_z, volq_high_x, volq_high_y, volq_high_z, volq, 
     & src_low_x, src_low_y, src_low_z, src_high_x, src_high_y, 
     & src_high_z, src)

      implicit none
      integer idxlo(3)
      integer idxhi(3)
      integer abskg_low_x, abskg_low_y, abskg_low_z, abskg_high_x, 
     & abskg_high_y, abskg_high_z
      double precision abskg(abskg_low_x:abskg_high_x, abskg_low_y:
     & abskg_high_y, abskg_low_z:abskg_high_z)
      integer esrcg_low_x, esrcg_low_y, esrcg_low_z, esrcg_high_x, 
     & esrcg_high_y, esrcg_high_z
      double precision esrcg(esrcg_low_x:esrcg_high_x, esrcg_low_y:
     & esrcg_high_y, esrcg_low_z:esrcg_high_z)
      integer volq_low_x, volq_low_y, volq_low_z, volq_high_x, 
     & volq_high_y, volq_high_z
      double precision volq(volq_low_x:volq_high_x, volq_low_y:
     & volq_high_y, volq_low_z:volq_high_z)
      integer src_low_x, src_low_y, src_low_z, src_high_x, src_high_y, 
     & src_high_z
      double precision src(src_low_x:src_high_x, src_low_y:src_high_y, 
     & src_low_z:src_high_z)
#endif /* __cplusplus */

#endif /* fspec_rdomsrc */

#ifndef PASS1
#define PASS1(x) x/**/_low, x/**/_high, x
#endif
#ifndef PASS3
#define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
