c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/clip_fort.h>
#else
      SUBROUTINE clip(indexLow, indexHigh, PCELL,
     &     STEP, X, N, FMAX, FMIN, IJ, IK, NIJK, IGMTKE, IGMF, 
     &     LCALF,LTEST,LKETM,
     &     ierr)
c     
c     This routine takes the step from UPDATE and clips it to make sure
C     that the addition of STEP to X (in GMRES) does not make X go out
c     of certain bounds.
c     
c     At present, the only variables that need this kind of correction
c     are f and tke.
c

      implicit none

#include "param4.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer PCELL(indexLow(1):indexHigh(1),indexLow(2):indexHigh(2),
     &     indexLow(3):indexHigh(3))
      double precision STEP(N)
      double precision X(N)
      integer N
      double precision FMAX, FMIN
      integer IK(indexLow(3):indexHigh(3))
      integer IJ(indexLow(2):indexHigh(2))
      integer NIJK, IGMTKE, IGMF
      logical LCALF,LTEST,LKETM
      integer ierr


c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer i, j, k, IJK
      integer NCOUNT

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)

c*********************************************************************
c     begin
c*********************************************************************
      NCOUNT = 0

      IF (LCALF) THEN
         DO 300 K=kBegGhost,kEndGhost
            DO 200 J=jBegGhost,jEndGhost
               DO 100 I=iBegGhost,iEndGhost
                  IJK = I + IJ(J) + IK(K) + (IGMF-1)*NIJK
                  IF (((X(IJK)+STEP(IJK)).LT.FMIN).AND. 
     &                 (PCELL(I,J,K).NE.WALL).AND.
     &                 .NOT.LTEST) THEN
                     STEP(IJK) = FMIN - X(IJK)
                     NCOUNT = NCOUNT + 1
                  ENDIF
                  IF (((X(IJK)+STEP(IJK)).GT.FMAX).AND. 
     &                 (PCELL(I,J,K).NE.WALL).AND.
     &                 .NOT.LTEST) THEN
                     STEP(IJK) = FMAX - X(IJK)
                     NCOUNT = NCOUNT + 1
                  ENDIF
 100           CONTINUE
 200        CONTINUE
 300     CONTINUE
      ENDIF

      IF (LKETM) THEN
         DO 600 K=kBegGhost,kEndGhost
            DO 500 J=jBegGhost,jEndGhost
               DO 400 I=iBegGhost,iEndGhost
                  IJK = I + IJ(J) + IK(K) + (IGMTKE-1)*NIJK
                  IF (((X(IJK)+STEP(IJK)).LT.SMALL).AND. 
     &                 (PCELL(I,J,K).NE.WALL).AND.
     &                 .NOT.LTEST) THEN
                     STEP(IJK) = SMALL - X(IJK)
                     NCOUNT = NCOUNT + 1
                  ENDIF
 400           CONTINUE
 500        CONTINUE
 600     CONTINUE
      ENDIF
      
      print*,'number of clippings performed = ',ncount
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

