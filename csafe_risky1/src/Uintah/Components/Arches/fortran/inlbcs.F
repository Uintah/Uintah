c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE inlbcs(domLoU, domHiU,  UU,
     &     domLoV, domHiV,  VV,
     &     domLoW, domHiW,  WW,
     &     domLo, domHi, idxLo, idxHi, DEN,
     &     PCELL, PFIELD,
     &     ierr)
C-----------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     This subroutine update inlet velocities in order to match
C          total flow rates while inlet area densities are changing.
C-----------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      double precision DEN(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer PFIELD
      integer ierr
      
c*********************************************************************
c     Locals :
c*********************************************************************
      integer iBegGhostU, iBegGhostV, iBegGhostW, iBegGhost
      integer iEndGhostU, iEndGhostV, iEndGhostW, iEndGhost
      integer jBegGhostU, jBegGhostV, jBegGhostW, jBegGhost
      integer jEndGhostU, jEndGhostV, jEndGhostW, jEndGhost
      integer kBegGhostU, kBegGhostV, kBegGhostW, kBegGhost
      integer kEndGhostU, kEndGhostV, kEndGhostW, kEndGhost
      integer ISTU, IENDU, JSTU, JENDU, KSTU, KENDU
      integer ISTV, IENDV, JSTV, JENDV, KSTV, KENDV
      integer ISTW, IENDW, JSTW, JENDW, KSTW, KENDW
      integer IST, IEND, JST, JEND, KST, KEND

      integer i, j, k

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhostU = domLoU(1)
      jBegGhostU = domLoU(2)
      kBegGhostU = domLoU(3)
      iEndGhostU = domHiU(1)
      jEndGhostU = domHiU(2)
      kEndGhostU = domHiU(3)
      iBegGhostV = domLoV(1)
      jBegGhostV = domLoV(2)
      kBegGhostV = domLoV(3)
      iEndGhostV = domHiV(1)
      jEndGhostV = domHiV(2)
      kEndGhostV = domHiV(3)
      iBegGhostW = domLoW(1)
      jBegGhostW = domLoW(2)
      kBegGhostW = domLoW(3)
      iEndGhostW = domHiW(1)
      jEndGhostW = domHiW(2)
      kEndGhostW = domHiW(3)
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

C-----------------------------------------------------------------------
C     EAST AND WEST INLETS
C-----------------------------------------------------------------------
      DO 120 K = kBegGhost,kEndGhost
         DO 100 J = jBegGhost,jEndGhost
            
            IF (PCELL(iBegGhost,J,K).EQ.PFIELD) THEN
               UU(iBegGhostU+1,J,K) = UU(iBegGhostU,J,K) *
     &              DEN(iBegGhost,J,K)
     &                 / (PT5*(DEN(iBegGhost,J,K) + DEN(IST,J,K)))
            ENDIF
            IF (PCELL(iEndGhost,J,K).EQ.PFIELD) THEN
               UU(iEndGhostU-1,J,K) = UU(iEndGhostU,J,K) *
     &              DEN(iEndGhost,J,K)
     &                 / (PT5*(DEN(iEndGhost,J,K) + DEN(IEND,J,K)))
            ENDIF
 100     CONTINUE
 120  CONTINUE
C--------------------------------------------------------------------
C     NORTH AND SOUTH INLETS
C--------------------------------------------------------------------
      DO 220 K = kBegGhost,kEndGhost
         DO 200 I = iBegGhost,iEndGhost
            
            IF (PCELL(I,jBegGhost,K).EQ.PFIELD) THEN
               VV(I,jBegGhostV+1,K) = VV(I,jBegGhostV,K) * 
     &              DEN(I,jBegGhost,K)
     &                 / (PT5*(DEN(I,jBegGhost,K) + DEN(I,JST,K)))
            ENDIF
            IF (PCELL(I,jEndGhost,K).EQ.PFIELD) THEN
               VV(I,jEndGhostV-1,K) = VV(I,jEndGhostV,K) * 
     &              DEN(I,jEndGhost,K)
     &                 / (PT5*(DEN(I,jEndGhost,K) + DEN(I,JEND,K)))
            ENDIF
 200     CONTINUE
 220  CONTINUE
C--------------------------------------------------------------------
C     TOP AND BOTTOM INLETS
C--------------------------------------------------------------------
      DO 320 J = jBegGhost,jEndGhost
         DO 300 I = iBegGhost,iEndGhost
            
            IF (PCELL(I,J,kBegGhost).EQ.PFIELD) THEN
               WW(I,J,kBegGhostW+1) = WW(I,J,kBegGhostW)
     &              * DEN(I,J,kBegGhost)
     &                 / (PT5*(DEN(I,J,kBegGhost) + DEN(I,J,KST)))
            ENDIF
            IF (PCELL(I,J,kEndGhost).EQ.PFIELD) THEN
               WW(I,J,kEndGhostW-1) = WW(I,J,kEndGhostW)
     &              * DEN(I,J,kEndGhost)
     &                 / (PT5*(DEN(I,J,kEndGhost) + DEN(I,J,KEND)))
            ENDIF
 300     CONTINUE
 320  CONTINUE
      
C--------------------------------------------------------------------
      
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.4  2000/07/07 23:07:47  rawat
c added inlet bc's
c
c Revision 1.3  2000/07/03 05:30:22  bbanerje
c Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
c
c Revision 1.2  2000/07/02 05:47:33  bbanerje
c Uncommented all PerPatch and CellInformation stuff.
c Updated array sizes in inlbcs.F
c
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
