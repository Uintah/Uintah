c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/mixltm_fort.h>
#else
      SUBROUTINE mixltm(indexLow, indexHigh, NITER, PCELL,
     &     UU, VV, WW,
     &     DEN, VIS,
     &     VISCOS,
     &     XX, YY, ZZ,
     &     SEW, SNS, STB,
     &     RR, 
     &     PRLS,
     &     CF, FAC_MESH, FILTERL,
     &     URFVIS,
     &     LWRAP, LTIM, LKETM,
     &     ierr)
C--------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE TURBULENT DIFFUSIVITY 
C          ACCORDING TO THE PRANDTL MIXING LENGTH MODEL.  THE 
C          MIXING LENGTH IS DETERMINED AS THE MINIMUM DISTANCE
C          AMONG THE REACTOR HEIGHT, LENGTH, AND DEPTH.
C--------------------------------------------------------------------

      implicit none

#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer NITER
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VIS(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VISCOS
      double precision XX(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision SEW(indexLow(1):indexHigh(1))
      double precision SNS(indexLow(2):indexHigh(2))
      double precision STB(indexLow(3):indexHigh(3))
      double precision RR(indexLow(2):indexHigh(2))
      double precision PRLS
      double precision CF, FAC_MESH, FILTERL
      double precision URFVIS
      logical LWRAP, LTIM, LKETM, LPRDAV, LPRDJK, LPRDKK
      integer ierr

c*********************************************************************
c     Locals
c*********************************************************************
      integer iBegGhost, iEndGhost, IST, IEND
      integer jBegGhost, jEndGhost, JST, JEND
      integer kBegGhost, kEndGhost, KST, KEND

      integer i, j, k
      double precision DE, DW, DN, DS, DT, DB, PMIXL, DMESH
      double precision UNP, USP, DUDY, UTP, UBP, DUDZ
      double precision VEP, VWP, DVDX, VTP, VBP, DVDZ
      double precision WEP, WWP, DWDX, WNP, WSP, DWDY
      double precision GRADVEL, CMLTM, VISOLD
      logical LVARML

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      IST = iBegGhost + 1
      JST = jBegGhost + 1
      KST = kBegGhost + 1
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)
      IEND = iEndGhost - 1
      JEND = jEndGhost - 1
      KEND = kEndGhost - 1

c*********************************************************************
c     Start
c*********************************************************************

      CMLTM = 0.09
      LVARML = .FALSE.
      DO 220 K = KST,KEND
         DO 210 J = JST,JEND
            DO 200 I = IST,IEND
C--------------------------------------------------------------------
C     CALCULATE MIXING OR FILTER LENGTH
C     THIS IS THE SMAGORINSKY MODEL WHEN DOING LES CALCULATIONS (LTIM)
C     so if you are using a Smagorinsky model you input the filter
C     width in the input file as PRLS.  Note the discussion in
C     Mason (1994) Q.J.R. Meteorol. Soc., 120, pp. 1-26
C     The filter length will become the cell size if it is very small
C--------------------------------------------------------------------
               IF (LTIM) THEN
                  DMESH = ((SNS(J)*SEW(I)*STB(K))**(1/3))
                  IF ((kEndGhost-kBegGhost).LE.3) 
     &                 DMESH = SQRT(SNS(J)*SEW(I))
                  IF ((jEndGhost-jBegGhost).LE.3) DMESH = 
     &                 SQRT(SEW(I)*STB(K))
                  IF ((iEndGhost-iBegGhost).LE.3) DMESH = 
     &                 SQRT(SNS(J)*STB(K))
                  PMIXL = CF*MAX(FILTERL,FAC_MESH*DMESH)
                  CMLTM = 1.0
               ELSE IF (LVARML) THEN
                  DW = XX(I) - XX(iBegGhost)
                  DE = XX(iEndGhost) - XX(I)
                  DS = YY(J) - YY(jBegGhost)
                  DN = YY(jEndGhost) - YY(J)
                  DB = ZZ(K) - ZZ(kBegGhost)
                  DT = ZZ(kEndGhost) - ZZ(K)
                  PMIXL = AMIN1(DW,DE,DS,DN,DB,DT)
               ELSE
                  PMIXL = PRLS/2.0
               END IF
C--------------------------------------------------------------------
C     CALCULATE GENERATION OF TURBULENCE
C--------------------------------------------------------------------
               UNP = (UU(I,J,K)+
     &              UU(I+1,J,K)+UU(I,J+1,K)+UU(I+1,J+1,K))/4.0
               USP = (UU(I,J,K)+
     &              UU(I+1,J,K)+UU(I,J-1,K)+UU(I+1,J-1,K))/4.0
               DUDY = (UNP-USP)/SNS(J)
               UTP = (UU(I,J,K)+
     &              UU(I+1,J,K)+UU(I,J,K+1)+UU(I+1,J,K+1))/4.0
               UBP = (UU(I,J,K)+
     &              UU(I+1,J,K)+UU(I,J,K-1)+UU(I+1,J,K-1))/4.0 
               DUDZ = (UTP-UBP)/(STB(K)*RR(J)) 
               VEP = (VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I+1,J,K)+VV(I+1,J+1,K))/4.0
               VWP = (VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I-1,J,K)+VV(I-1,J+1,K))/4.0
               DVDX = (VEP-VWP)/SEW(I)
               VTP = (VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I,J,K+1)+VV(I,J+1,K+1))/4.0
               VBP = (VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I,J,K-1)+VV(I,J+1,K-1))/4.0 
               DVDZ = (VTP-VBP)/(STB(K)*RR(J))
               WEP = (WW(I,J,K)+
     &              WW(I,J,K+1)+WW(I+1,J,K)+WW(I+1,J,K+1))/4.0
               WWP = (WW(I,J,K)+
     &              WW(I,J,K+1)+WW(I-1,J,K)+WW(I-1,J,K+1))/4.0
               DWDX = (WEP-WWP)/SEW(I)
               WNP = (WW(I,J,K)+
     &              WW(I,J,K+1)+WW(I,J+1,K)+WW(I,J+1,K+1))/4.0
               WSP = (WW(I,J,K)+
     &              WW(I,J,K+1)+WW(I,J-1,K)+WW(I,J-1,K+1))/4.0
               DWDY = (WNP-WSP)/SNS(J)
Cmjb... estimate gradient of velocity in transverse direction
Cmjb     * for a uni-axial flow, this would be "du/dy"
Cmjb       but must be generalized for 3D problems.
Cmjb     * open issue on what to use for the generalization 
Cmjb       or "norm" of the gradient term
Cmjb     * older versions of mixltm used a "norm" that 
Cmjb       could result in poor estimates for the turbulent 
Cmjb       portion of the viscosity in some situations.
Cmjb     * comparisons of a solution computed with k-e model for 
Cmjb       a simple duct flow and with using mixing length 
Cmjb       with different "norms" defined as
Cmjb           L0 =  (max{....}), 
Cmjb           L1 =  (sum of absolute value of terms),
Cmjb           L2 =  (square_root of the sum of the squares)
Cmjb       indicated better agreement in velocity field if used L2 norm.
Cmjb     * if the following causes problems, then would suggest 
Cmjb       trying either L0 or L1.
C            GRADVEL = ABS(DUDY + DUDZ + DVDX + DVDZ + DWDX + DWDY)

C            GRADVEL =SQRT( DUDY**2 + DUDZ**2 + DVDX**2 
C     &                   + DVDZ**2 + DWDX**2 + DWDY**2 )
               GRADVEL =SQRT(((DUDY + DVDX)**2) + ((DUDZ +
     &              DWDX)**2) + ((DVDZ + DWDY)**2))
Cmjb.... logic within next if/then block can be invoked
Cmjb     when using mixing length model to start up a k-e model
Cmjb      * logic is somewhat obtuse
Cmjb      * be sure logicals and "if statements" are set 
Cmjb        correctly for your problem.
               IF (PCELL(I,J,K).NE.WALL) THEN
                  VISOLD = VIS(I,J,K)
c              IF (LKETM) URFVIS = 0.01
c              IF (NITER.LT.5 .OR. .NOT.LKETM) THEN
                  VIS(I,J,K) = VISCOS + CMLTM*(PMIXL**2)*DEN(I,J,K)
     &                 *GRADVEL
c              END IF
                  IF (VISOLD.EQ.VISCOS) VISOLD = VIS(I,J,K)
                  VIS(I,J,K) = URFVIS*VIS(I,J,K)+(1.-URFVIS)*VISOLD
               END IF
 200        CONTINUE
 210     CONTINUE
 220  CONTINUE
C--------------------------------------------------------------------
C     PUT VIS(I,J,K) VALUES IN INLETS AND OUTLETS
C--------------------------------------------------------------------
      DO 410 K = kBegGhost,kEndGhost
        DO 400 J = jBegGhost,jEndGhost
           IF (PCELL(iBegGhost,J,K).NE.WALL) 
     &          VIS(iBegGhost,J,K) = VIS(IST,J,K)
           IF (PCELL(iEndGhost,J,K).NE.WALL) 
     &          VIS(iEndGhost,J,K) = VIS(IEND,J,K)
 400    CONTINUE
 410  CONTINUE
      DO 610 K = kBegGhost,kEndGhost
        DO 600 I = iBegGhost, iEndGhost
          IF (PCELL(I,jBegGhost,K).NE.WALL) 
     &          VIS(I,jBegGhost,K) = VIS(I,JST,K)
          IF (PCELL(I,jEndGhost,K).NE.WALL) 
     &          VIS(I,jEndGhost,K) = VIS(I,JEND,K)
 600   CONTINUE
 610  CONTINUE
      DO 810 J = jBegGhost, jEndGhost
         DO 800 I = iBegGhost, iEndGhost
            IF (PCELL(I,J,kBegGhost).NE.WALL) 
     &           VIS(I,J,kBegGhost) = VIS(I,J,KST)
            IF (PCELL(I,J,kEndGhost).NE.WALL) 
     &           VIS(I,J,kEndGhost) = VIS(I,J,KEND)
 800     CONTINUE
 810  CONTINUE
C--------------------------------------------------------------------
C     WRAP AROUND VIS(I,J,K) FOR CYLINDRICAL COORDINATE OPTION
C--------------------------------------------------------------------
      IF (LWRAP) 
     &     CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     &     VIS,RR, LPRDAV, LPRDJK, LPRDKK)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:48  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
