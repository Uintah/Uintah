c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/gminit_fort.h>
#else
      SUBROUTINE gminit(indexLow, indexHigh,
     &     UU, VV, WW, PP, FF, TED, TKE,
     &     UP, VP, WP, PT, FP, 
     &     XX,
     &     IJ, IK,
     &     NVAR,
     &     LSMPR, LGAS, LEXCP, LKETM, LCALF, LCALP,
     &     ierr)
C-----------------------------------------------------------------------
C     Written by Seshadri Kumar, January 26, 1997
C     This routine initializes the nonlinear GMRES work array, X
C     This is only done at the beginning of the nonlinear GMRES procedure.
C-----------------------------------------------------------------------

      implicit none

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision PP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FF(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TED(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision TKE(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision UP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision PT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision XX(*)
      integer IJ(indexLow(2):indexHigh(2))
      integer IK(indexLow(3):indexHigh(3))
      integer NVAR
      logical LSMPR, LGAS, LEXCP, LKETM, LCALF, LCALP
      integer ierr

c*********************************************************************
c     Locals :
c*********************************************************************
      integer IJKST
      integer iBegGhost, jBegGhost, kBegGHost
      integer iEndGhost, jEndGhost, kEndGHost
      
c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)

C-----EXECUTABLE STATEMENTS---------------------------------------------

      IJKST = 0
      
      IF (LGAS) THEN
         CALL READE(XX,UU,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost, IJ, IK)
         CALL READE(XX,VV,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost, IJ, IK)
         CALL READE(XX,WW,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost, IJ, IK)
         IF (
     $        (.NOT.LSMPR).OR.(LSMPR.AND..NOT.LEXCP)
     $        ) THEN
            CALL READE(XX,PP,NVAR,IJKST,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost, jEndGhost, kEndGhost,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost, jEndGhost, kEndGhost, IJ, IK)
         ENDIF
         IF (LKETM) THEN
            CALL READE(XX,TKE,NVAR,IJKST,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost, jEndGhost, kEndGhost,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost, jEndGhost, kEndGhost, IJ, IK)
            CALL READE(XX,TED,NVAR,IJKST,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost, jEndGhost, kEndGhost,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost, jEndGhost, kEndGhost, IJ, IK)
         ENDIF
         IF (LCALF) CALL READE(XX,FF,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost, IJ, IK)
      ENDIF

      IF (LCALP) THEN
         CALL READE(XX,UP,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost, IJ, IK)
         CALL READE(XX,VP,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost, IJ, IK)
         CALL READE(XX,WP,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost, IJ, IK)
         IF ((.NOT.LSMPR).OR.(LSMPR.AND..NOT.LEXCP)) THEN
            CALL READE(XX,FP,NVAR,IJKST,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost, jEndGhost, kEndGhost,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost, jEndGhost, kEndGhost, IJ, IK)
         ENDIF
         CALL READE(XX,PT,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost, jEndGhost, kEndGhost, IJ, IK)
      ENDIF

C-----------------------------------------------------------------------

      RETURN
      END

c*********************************************************************
c
c $Log $
c
c*********************************************************************

