c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/denaccum_fort.h>
c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
      DO 140 K=KST,KEND
         DO 130 J=JST,JEND
            DO 120 I=IST,IEND
               daccumulate = daccumulate + 
     $                       (den(i,j,k) - old_den(i,j,k))*
     $                       (sew(i)*sns(j)*stb(k))/deltat
 120           CONTINUE
 130        CONTINUE
 140     CONTINUE

      RETURN
      END

