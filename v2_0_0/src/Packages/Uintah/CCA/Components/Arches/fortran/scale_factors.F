c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/scale_factors_fort.h>
#else
      SUBROUTINE scale_factors(indexLow, indexHigh,
     $     UU, VV, WW, PP,
     &     TKE, TED, FF,
     &     UP,VP,WP,FP,PT,
     &     SCFI,SCFD,
     $     NVAR,
     $     LSMPR,LCALF,LKETM,LGAS,LCALP,LEXCP,
     &     IJ, IK,
     &     ierr)
*-----------------------------------------------------------------------
*     new module created by Seshadri Kumar (earlier part of evalf.f)
*     November 2, 1998
*
*     This module calculates the scale factors for all the variables.
*     Since, in the Newton iteration, we are operating on a 
*     preconditioned residual, of the form u-M(u) = B^inv*F(u),
*     the scaling factors for each component of the vector u are taken
*     to be average values of that component
*-----------------------------------------------------------------------
      implicit none

      integer indexLow(3), indexHigh(3)
      double precision
     $     UU(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     &     VV(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     &     WW(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     &     PP(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     $     TKE(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     &     TED(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     &     FF(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     $     UP(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     &     VP(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     &     WP(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     &     PT(indexLow(1):indexHigh(1),indexLow(2):indexhigh(2),
     &     indexLow(3):indexHigh(3)),
     $     FP(indexLow(1):indexHigh(1),indexLow(2):indexHigh(2),
     &     indexLow(3):indexHigh(3)),SCFI(*),SCFD(*)
      integer NVAR
      LOGICAL LSMPR,LCALF,LKETM,LGAS,LCALP,LEXCP
      integer IJ(indexLow(2):indexHigh(2))
      integer IK(indexLow(3):indexHigh(3))
      integer ierr

*     local variables

      integer iBegGhost, iEndGhost
      integer jBegGhost, jEndGhost
      integer kBegGhost, kEndGhost

      integer IJKST

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

C-----Calculate Scaling Factors-----------------------------------------

      IJKST = 0
      
      IF (LGAS) THEN
         CALL CALSCF(SCFI,UU,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost, IJ, IK)
         CALL CALSCF(SCFI,VV,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost, IJ, IK)
         CALL CALSCF(SCFI,WW,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost, IJ, IK)
         IF (
     $        (.NOT.LSMPR).OR.(LSMPR.AND..NOT.LEXCP)
     $        ) THEN
            CALL CALSCF(SCFI,PP,NVAR,IJKST,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost,jEndGhost,kEndGhost,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost,jEndGhost,kEndGhost, IJ, IK)
         ENDIF
         IF (LKETM) THEN
            CALL CALSCF(SCFI,TKE,NVAR,IJKST,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost,jEndGhost,kEndGhost,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost,jEndGhost,kEndGhost, IJ, IK)
            CALL CALSCF(SCFI,TED,NVAR,IJKST,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost,jEndGhost,kEndGhost,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost,jEndGhost,kEndGhost, IJ, IK)
         ENDIF
         IF (LCALF) CALL CALSCF(SCFI,FF,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost, IJ, IK)
      ENDIF
      
      IF (LCALP) THEN
         CALL CALSCF(SCFI,UP,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost, IJ, IK)
         CALL CALSCF(SCFI,VP,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost, IJ, IK)
         CALL CALSCF(SCFI,WP,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost, IJ, IK)
         IF (
     $        (.NOT.LSMPR).OR.(LSMPR.AND..NOT.LEXCP)
     $        ) THEN
            CALL CALSCF(SCFI,FP,NVAR,IJKST,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost,jEndGhost,kEndGhost,
     $           iBegGhost,jBegGhost,kBegGhost,
     &           iEndGhost,jEndGhost,kEndGhost, IJ, IK)
         ENDIF
         CALL CALSCF(SCFI,PT,NVAR,IJKST,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost,
     $        iBegGhost,jBegGhost,kBegGhost,
     &        iEndGhost,jEndGhost,kEndGhost, IJ, IK)
      ENDIF

c     debug, SK, July 20, 1998
c     copy scfi to scfd here
      CALL GCOPY(NVAR,SCFI,1,SCFD,1)
         
C-----------------------------------------------------------------------

      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:50  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

