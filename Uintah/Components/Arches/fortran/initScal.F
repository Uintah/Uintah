c*********************************************************************
c
c $Id$
c
c*********************************************************************
c GENERAL INFORMATION
c 
c    InitScal.F
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
c    Initializes scalars
c   
c WARNING
c    Other variables initialized in init.F
c   
c*********************************************************************

      SUBROUTINE initscal(domainLow, domainHigh, 
     &     indexLow, indexHigh, 
     &     SCALAR, SVAL)

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
      double precision SCALAR(domainLow(1):domainHigh(1),
     &     domainLow(2):domainHigh(2),
     &     domainLow(3):domainHigh(3))
      double precision SVAL

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
               SCALAR(I,J,K) = SVAL
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/06/13 19:52:58  bbanerje
c Initialize scalars
c
c
c*********************************************************************
