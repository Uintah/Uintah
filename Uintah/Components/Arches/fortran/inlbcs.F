c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE inlbcs(domLoU, domHiU, idxLoU, idxHiU, UU,
     &     domLoV, domHiV, idxLoV, idxHiV, VV,
     &     domLoW, domHiW, idxLoW, idxHiW, WW,
     &     domLo, domHi, idxLo, idxHi, DEN,
     &     PCELL, PFIELD,
     &     FLOWPR, AREAPR, DENPR,
c     &     UINLT,VINLT,WINLT,
c     &     INCELL,
c     &     LMODBC,
c     &     USAV, VSAV, WSAV,
     &     ierr)
C-----------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     This subroutine update inlet velocities in order to match
C          total flow rates while inlet area densities are changing.
C-----------------------------------------------------------------------

      implicit none

#include "param4.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3), idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3), idxLoV(3), idxHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3), idxLoW(3), idxHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      double precision DEN(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer PFIELD
      double precision FLOWPR, AREAPR, DENPR
c     double precision USAV(indexLow(1):indexHigh(1), 
c    &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
c     double precision VSAV(indexLow(1):indexHigh(1), 
c    &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
c     double precision WSAV(indexLow(1):indexHigh(1), 
c    &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
c     double precision UINLT(indexLow(2):indexHigh(2), 
c    &     indexLow(3):indexHigh(3))
c     double precision VINLT(indexLow(1):indexHigh(1), 
c    &     indexLow(3):indexHigh(3))
c     double precision WINLT(indexLow(1):indexHigh(1), 
c    &     indexLow(2):indexHigh(2))
c     integer INCELL(6, indexLow(1):indexHigh(1), 
c    &     indexLow(1):indexHigh(1))
c     logical LMODBC
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

C-----------------------------------------------------------------------
C     EAST AND WEST INLETS
C-----------------------------------------------------------------------
      DO 120 K = kBegGhost,kEndGhost
         DO 100 J = jBegGhost,jEndGhost
            
            IF (PCELL(iBegGhost,J,K).EQ.PFIELD .OR.
     &           PCELL(iBegGhost,J,K).EQ.SFIELD .OR.
     &           (PCELL(iBegGhost,J,K).EQ.DECOMP .AND. 
     $           UU(iBegGhost,J,K).GT.ZERO)) THEN
               IF (LMODBC) THEN
                  USAV(INCELL(1,J,K)+1,J,K) = 
     &                 UU(INCELL(1,J,K),J,K)
     &                 * DEN(INCELL(1,J,K),J,K)
     &                 / (PT5*(DEN(INCELL(1,J,K),J,K)+
     $                 DEN(INCELL(1,J,K)+1,J,K)))
               ENDIF
               UU(INCELL(1,J,K)+1,J,K) = 
     &              USAV(INCELL(1,J,K)+1,J,K)
            ENDIF
            
            IF (PCELL(iEndGhost,J,K).EQ.PFIELD .OR. 
     &           PCELL(iEndGhost,J,K).EQ.SFIELD .OR.
     &           (PCELL(iEndGhost,J,K).EQ.DECOMP .AND. 
     $           UINLT(J,K).LT.ZERO)) THEN
               IF (LMODBC) THEN
                  USAV(INCELL(2,J,K),J,K) = UINLT(J,K) * 
     $                 DEN(INCELL(2,J,K),J,K)
     &                 / (PT5*(DEN(INCELL(2,J,K),J,K)+
     $                 DEN(INCELL(2,J,K)-1,J,K)))
               ENDIF
               UU(INCELL(2,J,K),J,K) = 
     &              USAV(INCELL(2,J,K),J,K)
            ENDIF
            
 100     CONTINUE
 120  CONTINUE
C--------------------------------------------------------------------
C     NORTH AND SOUTH INLETS
C--------------------------------------------------------------------
      DO 220 K = kBegGhost,kEndGhost
         DO 200 I = iBegGhost,iEndGhost
            
            IF (PCELL(I,jBegGhost,K).EQ.PFIELD .OR.
     &           PCELL(I,jBegGhost,K).EQ.SFIELD .OR.
     &           (PCELL(I,jBegGhost,K).EQ.DECOMP .AND. 
     $           VV(I,jBegGhost,K).GT.ZERO)) THEN
               IF (LMODBC) THEN
                  VSAV(I,INCELL(3,I,K)+1,K) = 
     &                 VV(I,INCELL(3,I,K),K)
     &                 * DEN(I,INCELL(3,I,K),K)
     &                 / (PT5*(DEN(I,INCELL(3,I,K),K)+
     $                 DEN(I,INCELL(3,I,K)+1,K)))
               ENDIF
               VV(I,INCELL(3,I,K)+1,K) = 
     &                 VSAV(I,INCELL(3,I,K)+1,K)
            ENDIF
            
            IF (PCELL(I,jEndGhost,K).EQ.PFIELD .OR. 
     &           PCELL(I,jEndGhost,K).EQ.SFIELD .OR.
     &           (PCELL(I,jEndGhost,K).EQ.DECOMP .AND. 
     $           VINLT(I,K).LT.ZERO)) THEN
               IF (LMODBC) THEN
                  VSAV(I,INCELL(4,I,K),K) = VINLT(I,K) * 
     $                 DEN(I,INCELL(4,I,K),K)
     &                 / (PT5*(DEN(I,INCELL(4,I,K),K) + 
     $                 DEN(I,INCELL(4,I,K)-1,K)))
               ENDIF
               VV(I,INCELL(4,I,K),K) = 
     &              VSAV(I,INCELL(4,I,K),K)
            ENDIF
            
 200     CONTINUE
 220  CONTINUE
C--------------------------------------------------------------------
C     TOP AND BOTTOM INLETS
C--------------------------------------------------------------------
      DO 320 J = jBegGhost,jEndGhost
         DO 300 I = iBegGhost,iEndGhost
            
            IF (PCELL(I,J,kBegGhost).EQ.PFIELD .OR.
     &           PCELL(I,J,kBegGhost).EQ.SFIELD .OR.
     &           (PCELL(I,J,kBegGhost).EQ.DECOMP .AND. 
     $           WW(I,J,kBegGhost).GT.ZERO)) THEN
               IF (LMODBC) THEN
                  WSAV(I,J,INCELL(5,I,J)+1) = WW(I,J,INCELL(5,I,J)) 
     &                 * DEN(I,J,INCELL(5,I,J))
     &                 / (PT5*(DEN(I,J,INCELL(5,I,J)) + 
     $                 DEN(I,J,INCELL(5,I,J)+1)))
               ENDIF
               WW(I,J,INCELL(5,I,J)+1) = WSAV(I,J,INCELL(5,I,J)+1)
            ENDIF
            
            IF (PCELL(I,J,kEndGhost).EQ.PFIELD .OR.
     &           PCELL(I,J,kEndGhost).EQ.SFIELD .OR.
     &           (PCELL(I,J,kEndGhost).EQ.DECOMP .AND. 
     $           WINLT(I,J).LT.ZERO)) THEN
               IF (LMODBC) THEN
                  WSAV(I,J,INCELL(6,I,J)) = WINLT(I,J) * 
     $                 DEN(I,J,INCELL(6,I,J))
     &                 / (PT5*(DEN(I,J,INCELL(6,I,J)) + 
     $                 DEN(I,J,INCELL(6,I,J)-1)))
               ENDIF
               WW(I,J,INCELL(6,I,J)) = WSAV(I,J,INCELL(6,I,J))
            ENDIF
            
 300     CONTINUE
 320  CONTINUE
      
C--------------------------------------------------------------------
      
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.2  2000/07/02 05:47:33  bbanerje
c Uncommented all PerPatch and CellInformation stuff.
c Updated array sizes in inlbcs.F
c
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
