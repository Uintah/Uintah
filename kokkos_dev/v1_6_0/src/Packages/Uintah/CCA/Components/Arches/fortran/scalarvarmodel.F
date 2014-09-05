c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/scalarvarmodel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/param4.h>

c*********************************************************************
c     Locals
c*********************************************************************
      integer IST, IEND
      integer JST, JEND
      integer KST, KEND

      integer i, j, k
      double precision PMIXL, DMESH
      double precision scale, scalw, scaln, scals, scalt, scalb
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
               DMESH = ((SNS(J)*SEW(I)*STB(K))**(one/three))
               PMIXL = MAX(FILTERL,FAC_MESH*DMESH)
C--------------------------------------------------------------------
C     CALCULATE GENERATION OF TURBULENCE
C--------------------------------------------------------------------
               scale = pt5*(Scalar(i,j,k)+Scalar(i+1,j,k))
               scalw = pt5*(Scalar(i,j,k)+Scalar(i-1,j,k))
               scaln = pt5*(Scalar(i,j,k)+Scalar(i,j+1,k))
               scals = pt5*(Scalar(i,j,k)+Scalar(i,j-1,k))
               scalt = pt5*(Scalar(i,j,k)+Scalar(i,j,k+1))
               scalb = pt5*(Scalar(i,j,k)+Scalar(i,j,k-1))
               dfdx = (scale - scalw)/sew(I)
               dfdy = (scaln - scals)/sns(J)
               dfdz = (scalt - scalb)/stb(K)
               ScalarVar(I,J,K) = CFVar*(PMIXL**2)*
     $                            (dfdx**2 + dfdy**2 + dfdz**2)
 200        CONTINUE
 210     CONTINUE
 220  CONTINUE
      RETURN
      END

