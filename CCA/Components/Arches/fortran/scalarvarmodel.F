c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/scalarvarmodel_fort.h>

c*********************************************************************
c     Locals
c*********************************************************************
      integer IST, IEND
      integer JST, JEND
      integer KST, KEND

      integer i, j, k
      double precision PMIXL, DMESH
      double precision dfdx, dfdy, dfdz


c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

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
               PMIXL = MAX(FILTERL,FAC_MESH*DMESH)
C--------------------------------------------------------------------
C     CALCULATE GENERATION OF TURBULENCE
C--------------------------------------------------------------------
               dfdx = (Scalar(I,J,K) - Scalar(I-1,J,K))/dxpw(I)
               dfdy = (Scalar(I,J,K) - Scalar(I,J-1,K))/dyps(J)
               dfdz = (Scalar(I,J,K) - Scalar(I,J,K-1))/dzpb(K)
               ScalarVar(I,J,K) = CFVar*(PMIXL**2)*
     $                            (dfdx**2 + dfdy**2 + dfdz**2)
 200        CONTINUE
 210     CONTINUE
 220  CONTINUE
      RETURN
      END

