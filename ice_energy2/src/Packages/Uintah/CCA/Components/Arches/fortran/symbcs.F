c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/symbcs_fort.h>
#include "param4.h"
C--------------------------------------------------------------------
C     Subroutine provides symmtry boundary conditions for the 
C         following variables: p, pp, u, v, w, tke, and ted.
C-------------------------------------------------------------------- 

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k
      integer iBegGhost, iEndGhost, IST, IEND
      integer jBegGhost, jEndGhost, JST, JEND
      integer kBegGhost, kEndGhost, KST, KEND
      
c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = domLo(1)
      jBegGhost = domLo(2)
      kBegGhost = domLo(3)
      iEndGhost = domHi(1)
      jEndGhost = domHi(2)
      kEndGhost = domHi(3)
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

c*********************************************************************
c     Start :
c*********************************************************************
C$DIR SCALAR
      DO 420 K = kBegGhost,kEndGhost
C$DIR SCALAR
         DO 410 J = jBegGhost,jEndGhost
C$DIR SCALAR
            DO 400 I = iBegGhost,iEndGhost
C--------------------------------------------------------------------
C           SYMMETRY BOUNDARIES
C--------------------------------------------------------------------
               IF (PCELL(I,J,K).EQ.SYMTRY) THEN
                  IF (K.EQ.kBegGhost) THEN
                     IF (.NOT.(LPRDKK.OR.LPRDJK)) THEN
                        PHI(I,J,kBegGhost) = PHI(I,J,KST)
                        AB(I,J,KST) = ZERO
                     END IF
                  ELSE IF (K.EQ.kEndGhost) THEN
                     IF (.NOT.LPRDKK) THEN
                        PHI(I,J,kEndGhost) = PHI(I,J,KEND)
                        AT(I,J,KEND) = ZERO
                     END IF
                  ELSE IF (J.EQ.jBegGhost) THEN
                     IF (.NOT.LPRDJK) THEN
                        PHI(I,jBegGhost,K) = PHI(I,JST,K)
                        AS(I,JST,K) = ZERO
                     END IF
                  ELSE IF (J.EQ.jEndGhost) THEN
                     PHI(I,jEndGhost,K) = PHI(I,JEND,K)
                     AN(I,JEND,K) = ZERO
                  ELSE IF (I.EQ.iBegGhost) THEN
                     PHI(iBegGhost,J,K) = PHI(IST,J,K)
                     AW(IST,J,K) = ZERO
                  ELSE IF (I.EQ.iEndGhost) THEN
                     PHI(iEndGhost,J,K) = PHI(IEND,J,K)
                     AE(IEND,J,K) = ZERO
                  ELSE 
                     WRITE(6,*)' ERROR: SYMMETRY CONDITION 
     &                    SPECIFIED ON INTERIOR CELL '
                     WRITE(3,*)' ERROR: SYMMETRY CONDITION 
     &                    SPECIFIED ON INTERIOR CELL '
                  END IF
               END IF
 400        CONTINUE
 410     CONTINUE
 420  CONTINUE
      RETURN
      END

c*********************************************************************
c
c Revision 1.2  2000/07/13 04:51:35  bbanerje
c Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
c
c Revision 1.1  2000/04/12 20:31:51  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
