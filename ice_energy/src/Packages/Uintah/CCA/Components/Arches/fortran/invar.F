c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/invar_fort.h>
#else
      SUBROUTINE invar(indexLow, indexHigh, PCELL,
     &     UU,VV,WW,FF,
     &     FLOWIN,TKEIN,FMIXIN,
     &     SNS, STB,
     &     RA,
     &     FLODEN, FLOPPR, FLOPSC,
     &     PDEN, PNUPR, PNUSC,
     &     AREAPR, AREASC,
     &     FLOPIN, PTIN,
     &     SWNPR, SWNSC,
     &     FLOD, FLOWPR, FLOWSC,
     &     DENPR, DENSC,
     &     PRLS, SCLS,
     &     VISCOS,
     &     UP, VP, WP,
     &     DEN, DENP,
     &     TKE, TED, PT,
     &     LCALP, LRDPR, LCORD, LRSRT, LDBUG, LGAS, LKETM, LPRDJK,
     &     LSYMB, LCALF,
     &     ierr)
C-----------------------------------------------------------------------
C     INITIALIZATION ROUTINE FOR FLOWFIELD and SYMMETRY BOUNDARYS
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
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FF(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FLOWIN, TKEIN, FMIXIN
      double precision SNS(indexLow(2):indexHigh(2)) 
      double precision STB(indexLow(3):indexHigh(3)) 
      double precision RA(indexLow(2):indexHigh(2)) 
      double precision FLODEN, FLOPPR, FLOPSC
      double precision PDEN, PNUPR, PNUSC
      double precision AREAPR, AREASC
      double precision FLOPIN, PTIN
      double precision SWNPR, SWNSC
      double precision FLOD, FLOWPR, FLOWSC
      double precision DENPR, DENSC
      double precision PRLS, SCLS
      double precision VISCOS
      double precision UP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DENP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TKE(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TED(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision PT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      logical LCALP, LRDPR, LCORD, LRSRT, LDBUG, LGAS, LKETM, LPRDJK
      logical LSYMB, LCALF
      integer ierr

c*********************************************************************
c     Locals
c*********************************************************************
      integer iBegGhost, iEndGhost, iBegActual, iEndActual
      integer jBegGhost, jEndGhost, jBegActual, jEndActual
      integer kBegGhost, kEndGhost, kBegActual, kEndActual

      integer i, j, k, jk
      integer ICS
      double precision AVAREA, AVDENP, AVVELP, WVELP, AVPT, AVDEN
      double precision AVVEL, WVEL, AVLS, AVTKE, AVTED, REYGB

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
C     Get a representative cross-sectional area  
C-----------------------------------------------------------------------
      ICS = (iEndGhost+iBegGhost)/2
      AVAREA = ZERO
      DO 30 K = kBegActual,kEndActual
         DO 20 J = kBegActual,jEndActual
            IF (PCELL(ICS,J,K) .EQ. FFIELD) AVAREA = AVAREA + 
     &           SNS(J)*STB(K)*RA(J)
 20      CONTINUE
 30   CONTINUE
C-----------------------------------------------------------------------
C     SET NEEDED GLOBAL VARIABLES  
C-----------------------------------------------------------------------
c
c     calculate average particle density in domain: this step is needed 
c     first because computation of epsg can only then be done.
c     Note: at present I am assuming that we have a flow system in the 
c     particle phase (thus FLOPPR+FLOPSC .GT. 0) - will have to code in
c     different stuff later when this is not the case.
c
      IF (LCALP) THEN
c     the avdenp calculated here is the bulk density of the particles
         IF (LRDPR) THEN
            AVDENP = FLODEN/(FLOPPR+FLOPSC)
         ELSE
            AVDENP = PDEN*(FLOPPR*PNUPR+FLOPSC*PNUSC)/(FLOPPR+FLOPSC)
         ENDIF
         AVDENP = AVDENP*(AREAPR+AREASC)/AVAREA
         AVVELP= FLOPIN/(AVDENP*AVAREA)
         WVELP = ZERO
         IF (LCORD) WVELP = AVVELP*MAX(SWNPR,SWNSC)
c
c         AVPT  = AMIN1(ONE,PTIN)
c
c     am not sure why the amin1 exists: ask Packages/Phil!
c
         AVPT = PTIN
         IF (.NOT.LRSRT.AND.LDBUG) THEN 
            WRITE (6,*) '   UP Vel. re-initialized to: ',AVVELP,' m/s'
            WRITE (6,*) '     PT  re-initialized to: ',AVPT,' m2/s2'
         END IF
      ENDIF
c
      IF (LGAS) THEN
c
c         FM1 = ONE - FMIXIN
c
c         AVDEN = ONE/(FMIXIN/DENSF1 + FM1/DENSF0)
c     this procedure works only for cold flow.  
c
c
c      IF ((FLOWPR+FLOWSC).GT.ZERO) 
c     &         AVDEN = FLOWIN/(FLOWPR/DENPR + FLOWSC/DENSC)
c     this procedure does not work if lrdpr = true because then
c     denpr and densc are not properly defined (density is calculated
c     from f profiles read in from the .inl file) - in any case the 
c     definition of avden based on fmixin is more consistent with that 
c     used in prcf.f.  Remember, however, that we need to multiply avden
c     with an epsilon when particles are present.
c
         IF (LRDPR) THEN
            AVDEN = FLOD/(FLOWPR+FLOWSC)
         ELSE
            AVDEN = (FLOWPR*DENPR + FLOWSC*DENSC)/(FLOWPR+FLOWSC)
         ENDIF
         IF (LCALP) AVDEN = AVDEN*(AREAPR+AREASC)/AVAREA
         AVVEL = FLOWIN/(AVDEN*AVAREA)
         IF (LCORD) WVEL = AVVEL*MAX(SWNPR,SWNSC)
         AVLS = MAX(PRLS,SCLS)
         AVTKE = MIN(ONE,TKEIN)
         IF (LKETM) AVTED = EXP(LOG(AVTKE)*ONEPT5)*CMU/(0.03D0*AVLS)
         IF (VISCOS.NE.ZERO) THEN
            REYGB = AVDEN*AVVEL*SQRT(AVAREA)/VISCOS
            IF (LCORD) REYGB = AVDEN*AVVEL*SQRT(FOUR*AVAREA/PI)/
     $           VISCOS
            WRITE (6,*) '     Global Reynolds Number: ',REYGB
         ELSE
            WRITE (6,*) '  Inviscid Flow Calculation'
         END IF
c
c     modification for diffusion problems - currently hard-wired
c     - may need to have a logical later...
c     commented for general problems, SK, 04/23/95
c
c         AVDEN = ZERO
c         AVVEL = 1.0E-1
c
         IF (.NOT.LRSRT.AND.LDBUG) THEN 
            WRITE (6,*) '   U Vel. re-initialized to: ',AVVEL,' m/s'
            IF (LCALF) WRITE (6,*) '       f  re-initialized to: ',
     $           FMIXIN
            IF (LKETM) WRITE (6,*) 
     &           '     tke  re-initialized to: ',AVTKE,' m2/s2'
            IF (LKETM) WRITE (6,*) 
     &           '     ted  re-initialized to: ',AVTED,' m2/s3'
         END IF
      ENDIF
C-----------------------------------------------------------------------
C     SET SYMMETRY BOUNDARY CONDITIONS AND FLOW FIELD INITIAL GUESSES
C-----------------------------------------------------------------------
      IF (LPRDJK) THEN
         IF (LGAS) THEN
C$DIR SCALAR
            DO 60 JK = jBegGhost,jEndGhost
               IF (PCELL(iBegGhost,JK,kBegGhost).EQ.SYMTRY) 
     &              WW(iBegGhost,JK,kBegGhost) = 
     &              -VV(iBegGhost,jBegActual+1,JK)
               IF (PCELL(iBegGhost,jBegGhost,JK).EQ.SYMTRY) 
     &              WW(iBegGhost,jBegGhost,JK) = 
     &              VV(iBegGhost,JK,kBegActual)
               IF (PCELL(iBegGhost,JK,kBegGhost).EQ.SYMTRY) 
     &              VV(iBegGhost,JK,kBegGhost) = 
     &              WW(iBegGhost,jBegActual,JK)
               IF (PCELL(iBegGhost,jBegGhost,JK).EQ.SYMTRY) 
     &              VV(iBegGhost,jBegGhost,JK) = 
     &              -WW(iBegGhost,JK,kBegActual+1)
 60         CONTINUE
         ENDIF
         IF (LCALP) THEN
C$DIR SCALAR
            DO 61 JK = jBegGhost,jEndGhost
               IF (PCELL(iBegGhost,JK,kBegGhost).EQ.SYMTRY) 
     &              WP(iBegGhost,JK,kBegGhost) = 
     &              -VP(iBegGhost,jBegActual+1,JK)
               IF (PCELL(iBegGhost,jBegGhost,JK).EQ.SYMTRY) 
     &              WP(iBegGhost,jBegGhost,JK) = 
     &              VP(iBegGhost,JK,kBegActual)
               IF (PCELL(iBegGhost,JK,kBegGhost).EQ.SYMTRY) 
     &              VP(iBegGhost,JK,kBegGhost) = 
     &              WP(iBegGhost,jBegActual,JK)
               IF (PCELL(iBegGhost,jBegGhost,JK).EQ.SYMTRY) 
     &              VP(iBegGhost,jBegGhost,JK) = 
     &              -WP(iBegGhost,JK,kBegActual+1)
 61         CONTINUE
         ENDIF
      END IF 

      IF (.NOT.LRSRT) THEN
C$DIR SCALAR
         DO 120 K = kBegGhost,kEndGhost
C$DIR SCALAR
            DO 110 J = jBegGhost,jEndGhost
C$DIR SCALAR
               DO 100 I = iBegGhost,iEndGhost
                  IF (PCELL(I,J,K) .EQ. SYMTRY) LSYMB = .TRUE.
                  IF ((PCELL(I,J,K).EQ.FFIELD).OR.
     &                 (PCELL(I,J,K).EQ.SYMTRY)) THEN
                     DEN(I,J,K) = AVDEN
                     DENP(I,J,K) = AVDENP
                     IF ((I.GE.iBegActual).AND.
     &                    ((PCELL(I-1,J,K) .EQ. FFIELD).OR.
     &                    (PCELL(I-1,J,K) .EQ. SYMTRY))) THEN
                        UU(I,J,K) = AVVEL
C                        
C     this comment is activated for debug ONLY - for 2D channel
C     flow case!!!!!  sk, June 25, 1997 (we set all velocities in 
C     the domain to that at inlet - thus giving a fully developed 
C     initial guess throughout the domain - assuming, of course, that
C     u(1,J,K) is the exact f.d. vel. profile
C                        
C                        U(I,J,K) = U(1,J,K)
                        UP(I,J,K) = AVVELP
                     ENDIF
                     IF (LCORD) THEN
                        IF ((PCELL(I,J,K-1) .EQ. FFIELD) .OR.
     &                       (PCELL(I,J,K-1) .EQ. SYMTRY)) THEN
                           WW(I,J,K) = WVEL
                           WP(I,J,K) = WVELP
                        ENDIF
                     ENDIF
C
CINVISCID                     IF (PCELL(I-1,J,K) .EQ. FFIELD) THEN
C                        U(I,J,K) = U(I-1,J,K)
C                        UP(I,J,K) = UP(I-1,J,K)
C
                     IF (PCELL(I,J,K) .EQ. FFIELD .AND. LPRDJK) THEN
                        VV(I,J,K) = VV(iBegGhost,J,K)
                        WW(I,J,K) = WW(iBegGhost,J,K)
                        VP(I,J,K) = VP(iBegGhost,J,K)
                        WP(I,J,K) = WP(iBegGhost,J,K)
                     END IF
C                  IF (LCALF) F(I,J,K) = FMIXIN
                     FF(I,J,K) = FMIXIN
                     IF (LKETM) TKE(I,J,K) = AVTKE
                     IF (LKETM) TED(I,J,K) = AVTED
                     PT(I,J,K) = AVPT
                  END IF
 100           CONTINUE
 110        CONTINUE
 120     CONTINUE
      END IF
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
