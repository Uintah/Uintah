c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/outarea_fort.h>
#include "param4.h"

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k

c*********************************************************************
c     Start :
c*********************************************************************
      IST = indexLow(1)
      JST = indexLow(2)
      KST = indexLow(3)
      IEND = indexHigh(1)
      JEND = indexHigh(2)
      KEND = indexHigh(3)

C-------------------------------------------------------------------- 
C     CALCULATION OF THE AREA OF TOTAL OUTLET STREAMS
C--------------------------------------------------------------------
      if (xminus.or.xplus) then
         DO 110 K = KST,KEND
            DO 100 J = JST,JEND
               IF (xminus.and.PCELL(ist-1,J,K).EQ.PFIELD) THEN
                  AREAOUT=AREAOUT+pt5*(den(ist-1,j,k)+den(ist,j,k))
     $                      *SNS(J)*STB(K)
               END IF
               IF (xplus.and.PCELL(iend+1,J,K).EQ.PFIELD) THEN
                  AREAOUT = AREAOUT + SNS(J)*STB(K)*
     $                       pt5*(den(iend+1,j,k)+den(iend,j,k))
               END IF
 100        CONTINUE
 110     CONTINUE
      endif
      if (yminus.or.yplus) then
         DO 140 K = KST,KEND
            DO 130 I = IST,IEND
               IF (yminus.and.PCELL(I,jst-1,K).EQ.PFIELD) THEN
                  AREAOUT = AREAOUT + SEW(I)*STB(K)*
     $                 pt5*(den(i,jst,k)+den(i,jst-1,k))
               END IF
               IF (yplus.and.PCELL(I,jend+1,K).EQ.PFIELD) THEN
                  AREAOUT = AREAOUT + SEW(I)*STB(K)*
     $                 pt5*(den(i,jend,k)+den(i,jend+1,k))
               END IF
 130        CONTINUE
 140     CONTINUE
      endif

      if (zminus.or.zplus) then
         DO 170 J = JST,JEND
            DO 160 I = IST,IEND
               IF (zminus.and.PCELL(I,J,kst-1).EQ.PFIELD) THEN
                  AREAOUT = AREAOUT + SEW(I)*SNS(J)*
     $                 pt5*(den(i,j,kst)+den(i,j,kst+1))
               END IF
               IF (zplus.and.PCELL(I,J,kend+1).EQ.PFIELD) THEN
                  AREAOUT = AREAOUT + SEW(I)*SNS(J)*
     $                 pt5*(den(i,j,kend)+den(i,j,kend+1))
               END IF
 160        CONTINUE
 170     CONTINUE
      end if
      RETURN
      END
