c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/inketm_fort.h>
#else
      SUBROUTINE inketm(indexLow, indexHigh, PCELL,
     &     IJKEVL,
     &     UU,VV,WW,TKE,TED,
     &     SNS, RA,
     &     ierr)
C-----------------------------------------------------------------------
C     INITIALIZATION k-epsilon turbulence model
C     FOR FLOWFIELD and SYMMETRY BOUNDARYS
C   (this routine sets up initial guesses for the DEFAULT flowfield)   
C-----------------------------------------------------------------------
      
      implicit none

#include "param4.h"
#include "pconst.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      integer IJKEVL
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TED(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TKE(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision SNS(indexLow(2):indexHigh(2))
      double precision RA(indexLow(2):indexHigh(2))
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, iEndGhost, iBegActual, iEndActual
      integer jBegGhost, jEndGhost, jBegActual, jEndActual
      integer kBegGhost, kEndGhost, kBegActual, kEndActual

      integer ICS, KCS, i, j, k
      double precision CLS, TIN

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iBegActual = iBegGhost + 1
      jBegActual = jBegGhost + 1
      kBegActual = kBegGhost + 1
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)
      iEndActual = iEndGhost - 1
      jEndActual = jEndGhost - 1
      kEndActual = kEndGhost - 1

C-----------------------------------------------------------------------
C     Get a representative length scale  
C-----------------------------------------------------------------------
      ICS = MAX(iEndGhost/2,iBegActual)
      KCS = MAX(kEndGhost/2,kBegActual)
      DO 20 J = jBegGhost,jEndGhost
         IF (PCELL(ICS,J,KCS) .EQ. FFIELD) THEN
            CLS = SNS(J)*RA(J)
         END IF
 20   CONTINUE
      IF (CLS.LE.ZERO) THEN
         PRINT *, 'CLS turned up bad = ',CLS
         STOP
      END IF
C-----------------------------------------------------------------------
C     Set a default turbulence intensity
C-----------------------------------------------------------------------
      TIN = PT1
      WRITE (6,*) '               Initializing:  TKE and TED'
C-----------------------------------------------------------------------
C     SET SYMMETRY BOUNDARY CONDITIONS AND FLOW FIELD INITIAL GUESSES
C-----------------------------------------------------------------------
C$DIR SCALAR
      DO 120 K = kBegActual,kEndActual
C$DIR SCALAR
         DO 110 J = jBegActual,jEndActual
C$DIR SCALAR
            DO 100 I = iBegActual,iEndActual
               IF ((PCELL(I,J,K).EQ.FFIELD).OR.
     &              (PCELL(I,J,K).EQ.SYMTRY)) THEN
                  IF (IJKEVL.EQ.1) THEN
                     TKE(I,J,K) = ONEPT5*((PT5*(UU(I+1,J,K)+
     &                    UU(I,J,K)))**2
     &                    + VV(I,J,K)**2 + WW(I,J,K)**2) * (TIN**2)
                  ELSE IF (IJKEVL.EQ.2) THEN
                     TKE(I,J,K) = ONEPT5*((PT5*(VV(I,J+1,K)+
     &                    VV(I,J,K)))**2
     &                    + UU(I,J,K)**2 + WW(I,J,K)**2) * (TIN**2)
                  ELSE IF (IJKEVL.EQ.3) THEN
                     TKE(I,J,K) = ONEPT5*((PT5*(WW(I,J,K+1)+
     &                    WW(I,J,K)))**2
     &                    + VV(I,J,K)**2 + UU(I,J,K)**2) * (TIN**2)
                  END IF
                  IF (TKE(I,J,K).LE.ZERO) TKE(I,J,K) = 0.01D0
                  TED(I,J,K) = EXP(LOG(TKE(I,J,K))*ONEPT5)*
     $                 CMU/(0.03D0*CLS)
               ELSE IF (PCELL(I,J,K).EQ.WALL) THEN
                  TKE(I,J,K) = ZERO
                  TED(I,J,K) = ZERO
               END IF
 100        CONTINUE
 110     CONTINUE
 120  CONTINUE
C
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
