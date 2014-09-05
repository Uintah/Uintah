c*********************************************************************
c
c
c*********************************************************************
c GENERAL INFORMATION
c 
c    Init.F
c 
c    Author: Biswajit Banerjee (bbanerje@crsim.utah.edu)
c    University of Utah
c 
c    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
c   
c    Copyright (C) 2000 University of Utah
c 
c KEYWORDS
c    Arches Initialization
c 
c DESCRIPTION
c    Initializes [u,v,w]Velocity, pressure, density and viscosity
c   
c WARNING
c    Scalars initialized in initscal
c   
c*********************************************************************

      SUBROUTINE init(domLoU, domHiU, idxLoU, idxHiU, UU, UVAL,
     &     domLoV, domHiV, idxLoV, idxHiV, VV, VVAL,
     &     domLoW, domHiW, idxLoW, idxHiW, WW, WVAL,
     &     domLo, domHi, idxLo, idxHi, PP, PVAL,
     &     DEN, DENVAL,
     &     VIS, VISVAL)

C-----------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     INITIALIZATION ROUTINE     
C-----------------------------------------------------------------------

      implicit none

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3), idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2),
     &     domLoU(3):domHiU(3))
      double precision UVAL
      integer domLoV(3), domHiV(3), idxLoV(3), idxHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2),
     &     domLoV(3):domHiV(3))
      double precision VVAL
      integer domLoW(3), domHiW(3), idxLoW(3), idxHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2),
     &     domLoW(3):domHiW(3))
      double precision WVAL
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
      double precision PP(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      double precision PVAL
      double precision DEN(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      double precision DENVAL
      double precision VIS(domLo(1):domHi(1), domLo(2):domHi(2),
     &     domLo(3):domHi(3))
      double precision VISVAL

c*********************************************************************
c     Locals :
c*********************************************************************
      integer i, j, k
      integer iBegU, iEndU, iBegV, iBeg, iEndV, iBegW, iEndW, iEnd
      integer jBegU, jEndU, jBegV, jBeg, jEndV, jBegW, jEndW, jEnd
      integer kBegU, kEndU, kBegV, kBeg, kEndV, kBegW, kEndW, kEnd

c     integer tstIdx, tstdeli, tstdelj, tstdelk

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegU = idxLoU(1)
      jBegU = idxLoU(2)
      kBegU = idxLoU(3)
      iEndU = idxHiU(1)
      jEndU = idxHiU(2)
      kEndU = idxHiU(3)

      iBegV = idxLoV(1)
      jBegV = idxLoV(2)
      kBegV = idxLoV(3)
      iEndV = idxHiV(1)
      jEndV = idxHiV(2)
      kEndV = idxHiV(3)

      iBegW = idxLoW(1)
      jBegW = idxLoW(2)
      kBegW = idxLoW(3)
      iEndW = idxHiW(1)
      jEndW = idxHiW(2)
      kEndW = idxHiW(3)

      iBeg = idxLo(1)
      jBeg = idxLo(2)
      kBeg = idxLo(3)
      iEnd = idxHi(1)
      jEnd = idxHi(2)
      kEnd = idxHi(3)

C-----------------------------------------------------------------------
C     INITIALIZE ALL CARDINAL VARIABLES  
C-----------------------------------------------------------------------
      DO 3 K = kBegU,kEndU
         DO 2 J = jBegU,jEndU
            DO 1 I = iBegU,iEndU
               UU(I,J,K) = UVAL
 1          CONTINUE
 2       CONTINUE
 3    CONTINUE
      DO 30 K = kBegV,kEndV
         DO 20 J = jBegV,jEndV
            DO 10 I = iBegV,iEndV
               VV(I,J,K) = VVAL
 10         CONTINUE
 20      CONTINUE
 30   CONTINUE
      DO 31 K = kBegW,kEndW
         DO 21 J = jBegW,jEndW
            DO 11 I = iBegW,iEndW
               WW(I,J,K) = WVAL
 11         CONTINUE
 21      CONTINUE
 31   CONTINUE
      DO 300 K = kBeg,kEnd
         DO 200 J = jBeg,jEnd
            DO 100 I = iBeg,iEnd
               PP(I,J,K) = PVAL
               DEN(I,J,K) = DENVAL
               VIS(I,J,K) = VISVAL
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE

c*********************************************************************
c Test initialization
c*********************************************************************
c     tstdeli = iEndGhost-iBegGhost
c     tstdelj = jEndGhost-jBegGhost
c     tstdelk = kEndGhost-kBegGhost
c     DO 600 K = kBegGhost,kEndGhost
c        DO 500 J = jBegGhost,jEndGhost
c           DO 400 I = iBegGhost,iEndGhost
c              tstidx = tstdeli*tstdelj*(k-1) + tstdeli*(j-1) + (i-1)
c              UU(I,J,K) = dble(tstidx)
c              VV(I,J,K) = dble(tstidx)
c              WW(I,J,K) = dble(tstidx)
c              PP(I,J,K) = dble(tstidx)
c              DEN(I,J,K) = DENVAL
c              VIS(I,J,K) = VISVAL
c400        CONTINUE
c500     CONTINUE
c600  CONTINUE

c     print *, 'In Fortran : '
c     DO 900 K = kBegGhost,kEndGhost
c        DO 800 J = jBegGhost,jEndGhost
c           DO 700 I = iBegGhost,iEndGhost
c         write(*,'(3i4," : UU = ",f5.0," DEN = ",f5.0," VIS = ",f5.0)')
c    &              i, j, k, UU(i,j,k), DEN(i,j,k), VIS(i,j,k)
c700        CONTINUE
c800     CONTINUE
c900  CONTINUE
c    

      RETURN
      END

c*********************************************************************
c
c Revision 1.3  2000/06/28 08:14:55  bbanerje
c Changed the init routines a bit.
c
c Revision 1.2  2000/06/12 21:30:03  bbanerje
c Added first Fortran routines, added Stencil Matrix where needed,
c removed unnecessary CCVariables (e.g., sources etc.)
c
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
