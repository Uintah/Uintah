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
      double precision scalare, scalarw, scalarn
      double precision scalars, scalart, scalarb
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
               scalare = pt5*(Scalar(i,j,k)+Scalar(i+1,j,k))
               scalarw = pt5*(Scalar(i,j,k)+Scalar(i-1,j,k))
               scalarn = pt5*(Scalar(i,j,k)+Scalar(i,j+1,k))
               scalars = pt5*(Scalar(i,j,k)+Scalar(i,j-1,k))
               scalart = pt5*(Scalar(i,j,k)+Scalar(i,j,k+1))
               scalarb = pt5*(Scalar(i,j,k)+Scalar(i,j,k-1))
               dfdx = (scalare - scalarw)/sew(I)
               dfdy = (scalarn - scalars)/sns(J)
               dfdz = (scalart - scalarb)/stb(K)
               ScalarVar(I,J,K) = CFVar*(PMIXL**2)*
     $                            (dfdx**2 + dfdy**2 + dfdz**2)
 200        CONTINUE
 210     CONTINUE
 220  CONTINUE
      RETURN
      END

