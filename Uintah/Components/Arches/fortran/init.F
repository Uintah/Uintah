c*********************************************************************
c
c $Id$
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

      SUBROUTINE init(domainLow, domainHigh, 
     &     indexLow, indexHigh, 
     &     UU, UVAL,
     &     VV, VVAL,
     &     WW, WVAL,
     &     PP, PVAL,
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
      integer domainLow(3)
      integer domainHigh(3)
      integer indexLow(3)
      integer indexHigh(3)
      double precision UU(domainLow(1):domainHigh(1),
     &     domainLow(2):domainHigh(2),
     &     domainLow(3):domainHigh(3))
      double precision UVAL
      double precision VV(domainLow(1):domainHigh(1),
     &     domainLow(2):domainHigh(2),
     &     domainLow(3):domainHigh(3))
      double precision VVAL
      double precision WW(domainLow(1):domainHigh(1),
     &     domainLow(2):domainHigh(2),
     &     domainLow(3):domainHigh(3))
      double precision WVAL
      double precision PP(domainLow(1):domainHigh(1),
     &     domainLow(2):domainHigh(2),
     &     domainLow(3):domainHigh(3))
      double precision PVAL
      double precision DEN(domainLow(1):domainHigh(1),
     &     domainLow(2):domainHigh(2),
     &     domainLow(3):domainHigh(3))
      double precision DENVAL
      double precision VIS(domainLow(1):domainHigh(1),
     &     domainLow(2):domainHigh(2),
     &     domainLow(3):domainHigh(3))
      double precision VISVAL

c*********************************************************************
c     Locals :
c*********************************************************************
      integer i, j, k
      integer iBegGhost, iEndGhost, iBegActual, iEndActual
      integer jBegGhost, jEndGhost, jBegActual, jEndActual
      integer kBegGhost, kEndGhost, kBegActual, kEndActual

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = domainLow(1)
      jBegGhost = domainLow(2)
      kBegGhost = domainLow(3)
      iBegActual = indexLow(1)
      jBegActual = indexLow(2)
      kBegActual = indexLow(3)
      iEndGhost = domainHigh(1)
      jEndGhost = domainHigh(2)
      kEndGhost = domainHigh(3)
      iEndActual = indexHigh(1)
      jEndActual = indexHigh(2)
      kEndActual = indexHigh(3)

C-----------------------------------------------------------------------
C     INITIALIZE ALL CARDINAL VARIABLES  
C-----------------------------------------------------------------------
      DO 300 K = kBegGhost,kEndGhost
         DO 200 J = jBegGhost,jEndGhost
            DO 100 I = iBegGhost,iEndGhost
               UU(I,J,K) = UVAL
               VV(I,J,K) = VVAL
               WW(I,J,K) = WVAL
               PP(I,J,K) = PVAL
               DEN(I,J,K) = DENVAL
               VIS(I,J,K) = VISVAL
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.2  2000/06/12 21:30:03  bbanerje
c Added first Fortran routines, added Stencil Matrix where needed,
c removed unnecessary CCVariables (e.g., sources etc.)
c
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
