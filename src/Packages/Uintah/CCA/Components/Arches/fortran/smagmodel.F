c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/smagmodel_fort.h>

C--------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE TURBULENT DIFFUSIVITY 
C          ACCORDING TO THE PRANDTL MIXING LENGTH MODEL.  THE 
C          MIXING LENGTH IS DETERMINED AS THE MINIMUM DISTANCE
C          AMONG THE REACTOR HEIGHT, LENGTH, AND DEPTH.
C--------------------------------------------------------------------


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

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      IST = domLo(1)
      JST = domLo(2)
      KST = domLo(3)
      IEND = domHi(1)
      JEND = domHi(2)
      KEND = domHi(3)

c*********************************************************************
c     Start
c*********************************************************************
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
               DMESH = ((SNS(J)*SEW(I)*STB(K))**(1/3))
               PMIXL = CF*MAX(FILTERL,FAC_MESH*DMESH)
               CMLTM = 1.0
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
               DUDZ = (UTP-UBP)/STB(K) 
               VEP = (VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I+1,J,K)+VV(I+1,J+1,K))/4.0
               VWP = (VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I-1,J,K)+VV(I-1,J+1,K))/4.0
               DVDX = (VEP-VWP)/SEW(I)
               VTP = (VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I,J,K+1)+VV(I,J+1,K+1))/4.0
               VBP = (VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I,J,K-1)+VV(I,J+1,K-1))/4.0 
               DVDZ = (VTP-VBP)/STB(K)
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
               VISOLD = VIS(I,J,K)
c              IF (LKETM) URFVIS = 0.01
c              IF (NITER.LT.5 .OR. .NOT.LKETM) THEN
               VIS(I,J,K) = VISCOS + CMLTM*(PMIXL**2)*DEN(I,J,K)
     &              *GRADVEL
c              END IF
 200        CONTINUE
 210     CONTINUE
 220  CONTINUE
      RETURN
      END

c*********************************************************************
c
c Revision 1.4  2000/10/12 00:03:24  rawat
c running for more than one timestep.
c
c Revision 1.3  2000/07/01 05:21:01  bbanerje
c Changed CellInformation calcs for Turbulence model requirements ..
c CellInformation still needs work.
c
c Revision 1.2  2000/06/30 05:12:20  bbanerje
c Changed reComputeTurbModel to reflect chnages to computeTurbModel.
c Changed name Subroutine mixltm to Subroutine smagmodel
c
c Revision 1.1  2000/06/30 04:19:18  rawat
c added turbulence model and compute properties
c
c Revision 1.1  2000/04/12 20:31:48  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
