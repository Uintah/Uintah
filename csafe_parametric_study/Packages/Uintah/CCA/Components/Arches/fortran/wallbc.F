c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/wallbc_fort.h>
#include "param4.h"

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i,j,k
      integer ist,jst,kst,iend,jend,kend

c*********************************************************************
c     Start :
c*********************************************************************
      ist = idxLo(1)
      jst = idxLo(2)
      kst = idxLo(3)
      iend = idxHi(1)
      jend = idxHi(2)
      kend = idxHi(3)

c WARNING! Only xminus wall BC is supported
      if (xminus) then
         DO 500 K = kst,kend
            DO 400 J = jst,jend
               I = ist -1
               IF (PCELL(I,J,K).EQ.WALL) THEN
                  IF (PCELL(I+1,J,K).NE.WALL) 
     $                 AW(I+1,J,K) = ZERO
               END IF
 400        CONTINUE
 500     CONTINUE
      endif
      RETURN
      END 
