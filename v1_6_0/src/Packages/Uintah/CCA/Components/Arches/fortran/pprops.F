c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/pprops_fort.h>
#else
       SUBROUTINE pprops(indexLow, indexHigh, PCELL,
     &     FP, PT,
     &     DENP, VISP,
     &     RR,
     &     XX, YY, ZZ,
     &     VISCOS, PDEN, PDIA, ALP, CRES,
     &     URDENP, URVISP,
     &     FPFIX, TFIX, 
     &     INITPR,
     &     IFPREF, JFPREF, KFPREF,
     &     NITER,
     &     LTEST, LWRAP, LPRDAV, LPRDJK, LPRDKK,
     &     ierr)
C--------------------------------------------------------------------
C     This subroutine calculates particle properties like the density
C     and viscosity from the calculated particle pressure
C--------------------------------------------------------------------

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
      double precision FP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision PT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DENP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VISP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision RR(indexLow(2):indexHigh(2))
      double precision XX(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision VISCOS, PDEN, PDIA, ALP, CRES
      double precision URDENP, URVISP
      double precision FPFIX, TFIX
      integer INITPR
      integer IFPREF, JFPREF, KFPREF
      integer NITER
      logical LTEST, LWRAP, LPRDKK, LPRDAV, LPRDJK
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, iEndGhost, iBegActual, iEndActual
      integer jBegGhost, jEndGhost, jBegActual, jEndActual
      integer kBegGhost, kEndGhost, kBegActual, kEndActual

      integer i, j, k
      double precision PNUMAX, PI, PT33, PNU, GONU
      double precision PNULIM, GNUL, RLLIM
      double precision DENOLD, DENNEW, VISOLD, VISNEW
      double precision RLAMB1, RLAMB2

c*********************************************************************
c     External Functions :
c*********************************************************************
      double precision root

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

c*********************************************************************
c     Start :
c*********************************************************************
      PNUMAX = 0.6436D0
c     PI = FOUR*ATAN(ONE)
      PT33 = ONE/THREE

*******************************************************************
      IF (LTEST) THEN
C
C     I am coding in a simple equation of state for the particles:
C     P = denp*T*C
C     of course, since in iopt=4 for the exact solution we have
C     p = denp*xyz and T=z, implying C=xy
C     
C     If ltest then we specify the density at the start as the exact 
C     value and offset the velocities and perhaps the pressure and let
C     the solution converge to the exact values.
C     note: if ltest fpfix is defined in test.f 
C
         DO K=kBegActual,kEndActual
            DO J=jBegActual,jEndActual
               DO I=iBegActual,iEndActual
                  DENOLD = DENP(I,J,K)
                  DENNEW = FP(I,J,K)/(PT(I,J,K)*XX(I)*YY(J))
                  DENP(I,J,K) = URDENP*DENNEW + (ONE-URDENP)*DENOLD
               END DO
            END DO
         END DO
      ELSE
c     
c     if not ltest then we specify the initial pressure from the guessed 
c     density field before the first iteration, based on our eqn. of state.
c     for niter>1, we back-calculate the density from the pressure.
c
         IF (NITER.EQ.0) THEN
            PT(iBegActual,jBegActual,kBegActual) = 
     &           PT(iBegGhost,jBegActual,kBegActual)
            TFIX = PT(iBegActual,jBegActual,kBegActual)
            DENP(iBegActual,jBegActual,kBegActual) = 
     &           DENP(iBegGhost,jBegActual,kBegActual)
            DO 290 K = kBegGhost,kEndGhost
               DO 280 J = jBegGhost,jEndGhost
                  DO 270 I = iBegGhost,iEndGhost
                     IF (PCELL(I,J,K).NE.WALL) THEN
                        PNU = DENP(I,J,K)/PDEN
c                        GONU = PT5*(TWO-PNU)/(1-PNU)**3
                        GONU = PT6/(ONE-(PNU/PNUMAX)**PT33)
                        FP(I,J,K) = TWO*PDEN*PNU*PNU*GONU*(ONE+CRES)*
     &                              PT(I,J,K)
                        FP(I,J,K) = FP(I,J,K) + DENP(I,J,K)*PT(I,J,K)
                        VISP(I,J,K) = PT4*(TWO+ALP)*PDEN*PNU*PNU*GONU*
     &                       PDIA*(ONE+CRES)*(PT(I,J,K)/PI)**PT5
c     override: debug
                        VISP(I,J,K) = VISCOS
                     ENDIF
 270              CONTINUE
 280           CONTINUE
 290        CONTINUE
            IF (LWRAP) CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &           VISP,RR,LPRDAV, LPRDJK, LPRDKK)
            IF (LWRAP) CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &           FP,RR,LPRDAV, LPRDJK, LPRDKK)
c
c     calculation of fpfix and tfix: currently this is hard-wired without any
c     logic - i have just set the particle reference pressure location as
c     one i node downstream of the primary inlet at (1,5,5) - later I may need
c     to put in some logic where I scan the geometry for pfield/sfield and set
c     the first cell downstream of the first pcell I find as the reference location
c     (provided that cell is a ffield) ... but that's for later.
c     Also, I'm now also keeping pt at the ref.node equal to that of the pfield
c     cell so that we will have the same denp at the reference location as at the
c     pfield cell.
c     SK, 3/30/95
c
            FPFIX = FP(IFPREF,JFPREF,KFPREF)
c
c     limit calculations for niter>1 
c
            PNULIM = PNUMAX
            GNUL = PT6/(ONE-(PNULIM/PNUMAX)**PT33)
            RLLIM = FOUR*PNULIM**2*GNUL + PNULIM
c
c     rllim is the maximum possible value of fp/(pden*pt) from the
c     D&G model
c
         ELSE IF (NITER.GE.INITPR.AND..NOT.LTEST) THEN
            DO 190 K = kBegGhost,kEndGhost
               DO 180 J = kBegGhost,jEndGhost
                  DO 170 I = iBegGhost,iEndGhost
                     IF (PCELL(I,J,K).NE.WALL .AND. PCELL(I,J,K).NE.
     &                    PFIELD .AND. PCELL(I,J,K).NE.SFIELD) THEN
                        DENOLD = DENP(I,J,K)
c
c     based on Ding and Gidaspow (1990) model [(A) and (C)]
c
                        RLAMB1 = FP(I,J,K)/(PDEN*PT(I,J,K))
                        RLAMB2 = TWO*(ONE+CRES)*PT6
                        IF (RLAMB1.LE.RLLIM) THEN
                           PNU = ROOT(ZERO,ONE,PT1,0.00001D0,RLAMB1,
     &                                RLAMB2)
                        ELSE
                           PNU = PNUMAX
                        ENDIF
                        DENNEW = PNU*PDEN
                        DENP(I,J,K) = URDENP*DENNEW+(ONE-URDENP)*DENOLD
                        VISOLD = VISP(I,J,K)
                        PNU = DENP(I,J,K)/PDEN
                        GONU = PT6/(ONE-(PNU/PNUMAX)**PT33)
                        VISNEW = PT4*(TWO+ALP)*PDEN*PNU**2*GONU
     &                       *PDIA*(ONE+CRES)*(PT(I,J,K)/PI)**PT5
                        VISP(I,J,K) = URVISP*VISNEW+(ONE-URVISP)*VISOLD
                     END IF
 170              CONTINUE
 180           CONTINUE
 190        CONTINUE
            IF (LWRAP) CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &           VISP,RR,LPRDAV, LPRDJK, LPRDKK)
            IF (LWRAP) CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &           DENP,RR,LPRDAV, LPRDJK, LPRDKK)
         END IF
      END IF
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
