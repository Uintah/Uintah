c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE inlbcs(indexLow, indexHigh, PCELL,
     &     UU,VV,WW, 
     &     DEN,
     &     UINLT,VINLT,WINLT,
     &     INCELL,
     &     LMODBC,
     &     USAV, VSAV, WSAV,
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
      integer indexLow(3)
      integer indexHigh(3)
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision USAV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VSAV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WSAV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UINLT(indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision VINLT(indexLow(1):indexHigh(1), 
     &     indexLow(3):indexHigh(3))
      double precision WINLT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2))
      integer INCELL(6, indexLow(1):indexHigh(1), 
     &     indexLow(1):indexHigh(1))
      logical LMODBC
      integer ierr
      
c*********************************************************************
c     Locals :
c*********************************************************************
      integer          iBegGhost, iEndGhost
      integer          jBegGhost, jEndGhost
      integer          kBegGhost, kEndGhost

      integer i, j, k

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)

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
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
