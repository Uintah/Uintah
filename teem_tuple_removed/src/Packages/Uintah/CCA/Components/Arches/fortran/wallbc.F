c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/wallbc_fort.h>
#include "param4.h"

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost,jBegGhost,kBegGhost
      integer iEndGhost,jEndGhost,kEndGhost,i,j,k
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
      if (xminus) then
         iBegGhost = ist -1
      else
         iBegGhost = ist
      endif
      if (yminus) then
         jBegGhost = jst -1
      else
         jBegGhost = jst
      endif
      if (zminus) then
         kBegGhost = kst - 1
      else
         kBegGhost = kst
      endif
      if (xplus) then
         iEndGhost = iend +1
      else
         iEndGhost = iend
      endif
      if (yplus) then
         jEndGhost = jend + 1
      else
         jEndGhost = jend
      endif
      if (zplus) then
         kEndGhost = kend+1
      else
         kEndGhost = kend
      endif

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

c*********************************************************************
c
c Revision 1.5  2000/10/07 23:53:03  rawat
c modified bc subroutines
c
c Revision 1.4  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.3  2000/09/26 19:59:19  sparker
c Work on MPI petsc
c
c Revision 1.2  2000/07/13 04:51:35  bbanerje
c Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
c
c Revision 1.1  2000/04/12 20:31:51  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
