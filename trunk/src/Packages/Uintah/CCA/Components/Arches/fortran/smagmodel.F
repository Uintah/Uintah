c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/smagmodel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/param4.h>


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
      double precision UEP, UWP, VNP, VSP, WTP, WBP
      double precision GRADVEL, CMLTM, VISOLD
      double precision s11,s22,s33,s12,s13,s23,IsI

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
               DMESH = ((SNS(J)*SEW(I)*STB(K))**(1.0D0/3.0D0))
               PMIXL = CF*MAX(FILTERL,FAC_MESH*DMESH)
               CMLTM = 1.0D0
C--------------------------------------------------------------------
C     CALCULATE GENERATION OF TURBULENCE
C--------------------------------------------------------------------
               uep = uu(i+1,j,k)
               uwp = uu(i,j,k)
               UNP = 0.25D0*(UU(I,J,K)+
     &              UU(I+1,J,K)+UU(I,J+1,K)+UU(I+1,J+1,K))
               USP = 0.25D0*(UU(I,J,K)+
     &              UU(I+1,J,K)+UU(I,J-1,K)+UU(I+1,J-1,K))
               DUDY = (UNP-USP)/SNS(J)
               UTP = 0.25D0*(UU(I,J,K)+
     &              UU(I+1,J,K)+UU(I,J,K+1)+UU(I+1,J,K+1))
               UBP = 0.25D0*(UU(I,J,K)+
     &              UU(I+1,J,K)+UU(I,J,K-1)+UU(I+1,J,K-1))
               DUDZ = (UTP-UBP)/STB(K) 
               vnp = vv(i,j+1,k)
               vsp = vv(i,j,k)
               VEP = 0.25D0*(VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I+1,J,K)+VV(I+1,J+1,K))
               VWP = 0.25D0*(VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I-1,J,K)+VV(I-1,J+1,K))
               DVDX = (VEP-VWP)/SEW(I)
               VTP = 0.25D0*(VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I,J,K+1)+VV(I,J+1,K+1))
               VBP = 0.25D0*(VV(I,J,K)+
     &              VV(I,J+1,K)+VV(I,J,K-1)+VV(I,J+1,K-1)) 
               DVDZ = (VTP-VBP)/STB(K)
               wtp = ww(i,j,k+1)
               wbp = ww(i,j,k)
               WEP = 0.25D0*(WW(I,J,K)+
     &              WW(I,J,K+1)+WW(I+1,J,K)+WW(I+1,J,K+1))
               WWP = 0.25D0*(WW(I,J,K)+
     &              WW(I,J,K+1)+WW(I-1,J,K)+WW(I-1,J,K+1))
               DWDX = (WEP-WWP)/SEW(I)
               WNP = 0.25D0*(WW(I,J,K)+
     &              WW(I,J,K+1)+WW(I,J+1,K)+WW(I,J+1,K+1))
               WSP = 0.25D0*(WW(I,J,K)+
     &              WW(I,J,K+1)+WW(I,J-1,K)+WW(I,J-1,K+1))
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
c              calculate the grid strain rate tensor

               s11 = (uep-uwp)/sew(i)
               s22 = (vnp-vsp)/sns(j)
               s33 = (wtp-wbp)/stb(k)
               s12 = 0.5D0*((unp-usp)/sns(j) + (vep-vwp)/sew(i))
               s13 = 0.5D0*((utp-ubp)/stb(k) + (wep-wwp)/sew(i))
               s23 = 0.5D0*((vtp-vbp)/stb(k) + (wnp-wsp)/sns(j))

c              calculate absolute value of the grid strain rate
   
               IsI = sqrt(2.0D0*(s11**2 + s22**2 + s33**2 +
     &              2.0D0*(s12**2 + s13**2 + s23**2)))

Cmjb.... logic within next if/then block can be invoked
Cmjb     when using mixing length model to start up a k-e model
Cmjb      * logic is somewhat obtuse
Cmjb      * be sure logicals and "if statements" are set 
Cmjb        correctly for your problem.
               VISOLD = VIS(I,J,K)
c              IF (LKETM) URFVIS = 0.01
c              IF (NITER.LT.5 .OR. .NOT.LKETM) THEN
c               if (I.lt.10) then
c                  vis(i,j,k) = viscos
c               else
                  VIS(I,J,K) = VISCOS + CMLTM*(PMIXL**2)*DEN(I,J,K)
     &                        *IsI
c     $                         *GRADVEL
c               endif
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
