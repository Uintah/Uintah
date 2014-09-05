c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE scalarvarmodel(domLoSc, domHiSc, Scalar,
     &     domLoScVar, domHiScVar,
     &     idxLo, idxHi, ScalarVar,
     $     domLo, domHi,
     &     dxpw, dyps, dzpb,
     &     SEW, SNS, STB,
     &     CFVar, FAC_MESH, FILTERL,
     &     ierr)
C--------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE TURBULENT DIFFUSIVITY 
C          ACCORDING TO THE PRANDTL MIXING LENGTH MODEL.  THE 
C          MIXING LENGTH IS DETERMINED AS THE MINIMUM DISTANCE
C          AMONG THE REACTOR HEIGHT, LENGTH, AND DEPTH.
C--------------------------------------------------------------------

      implicit none


c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoSc(3)
      integer domHiSc(3)
      integer domLoScVar(3)
      integer domHiScVar(3)
      integer domLo(3)
      integer domHi(3)
      integer idxLo(3)
      integer idxHi(3)
      double precision Scalar(domLoSc(1):domHiSc(1), 
     &     domLoSc(2):domHiSc(2),domLoSc(3):domHiSc(3))
      double precision ScalarVar(domLoScVar(1):domHiScVar(1), 
     &     domLoScVar(2):domHiScVar(2),domLoScVar(3):domHiScVar(3))
      double precision dxpw(domLo(1):domHi(1))
      double precision dyps(domLo(2):domHi(2))
      double precision dzpb(domLo(3):domHi(3))
      double precision SEW(domLo(1):domHi(1))
      double precision SNS(domLo(2):domHi(2))
      double precision STB(domLo(3):domHi(3))
      double precision CFVar, FAC_MESH, FILTERL
      integer ierr

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

