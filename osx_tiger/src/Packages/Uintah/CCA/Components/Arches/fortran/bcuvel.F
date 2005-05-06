c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/bcuvel_fort.h>
#include "param4.h"
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision const
      integer IST, JST, KST, IEND, JEND, KEND
      integer IBegGhost, JBegGhost, KBegGhost
      integer IEndGhost, JEndGhost, KEndGhost
      integer idxUInlet, idxVInlet, idxWInlet
      integer i, j, k

      double precision YP, ZP, TMULT

c*******************u*************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
      if (xminus) then
         iBegGhost = ist-1
      else
         iBegGhost = ist
      endif
      if (yminus) then
         jBegGhost = jst-1
      else
         jBegGhost = jst
      endif
      if (zminus) then
         kBegGhost = kst - 1
      else
         kBegGhost = kst
      endif
      if (xplus) then
         iEndGhost = iend+1
      else
         iEndGhost = iend
      endif
      if (yplus) then
         jEndGhost = jend+1
      else
         jEndGhost = jend
      endif
      if (zplus) then
         kEndGhost = kend + 1
      else
         kEndGhost = kend
      endif
      
c------------------------------------------------------------------ 
c     interior intrusions
c------------------------------------------------------------------ 
      const = ONE
      do 820 k=kst, kend
         do 810 j=jst, jend
            do 800 i=ist, iend

               if (pcell(i-1,j,k).eq.wall.or. pcell(i,j,k).eq.wall)
     $              call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     $                          as(i,j,k),at(i,j,k),ab(i,j,k),
     $                          su(i,j,k),sp(i,j,k),const)

 800        continue
 810     continue
 820  continue


c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      if (xminus) then
C$DIR SCALAR
         DO 120 K = kst, kend
C$DIR SCALAR
            DO 110 J = jst, jend
C$DIR SCALAR
               I = ist -1
               if (pcell(I,J,K).eq.wall) then
c                  uu(i,j,k) = 0.0D0
C--------------------------------------------------------------------
C             East FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                  IF (PCELL(I+1, J, K).NE.WALL) THEN
                     AW(I+2,J,K) = 0.0D0
                     cwsav(i+2,j,k) = 0.0D0
                  end if
               end if
 110        CONTINUE
 120     CONTINUE
      endif

C--------------------------------------------------------------------
C     PRESSURE BOUNDARY CONDITIONS 
C--------------------------------------------------------------------
C--------------------------------------------------------------------
C     East AND/OR West WALLS
C--------------------------------------------------------------------
      DO 310 K = kst, kend
        DO 300 J = jst, jend
           IF ((PCELL(iEndGhost, J, K).EQ.PBCFLD).or.
     $          (PCELL(iEndGhost, J, K).EQ.outletfld)) THEN
c              UU(iEndGhost+1,J,K) = UU(iEndGhost,J,K)
c              AE(iend,J,K) = 2.0D0*TINY
          END IF
          IF (PCELL(iBegGhost,J,K).EQ.PBCFLD) THEN
c             UU(iBegGhost,J,K) = UU(ist,J,K)
          END IF
  300   CONTINUE
  310 CONTINUE
C--------------------------------------------------------------------
C     North AND/OR South WALLS
C--------------------------------------------------------------------
      DO 410 K = kst, kend
        DO 400 I = ist, iend
          IF ((PCELL(I,jEndGhost, K).EQ.PBCFLD)) then
c .AND.
c     $        (VV(I,jEndGhost+1,K).GT.0.0D0)) THEN
c             UU(I,jEndGhost,K) = UU(I,jend,K)
c             AN(I,jend,K) = 2.0D0*TINY
c             UU(I,jEndGhost,K) = UU(I,jend,K)
c             AN(I,jend,K) = 2.0D0*TINY
          END IF
          IF (PCELL(I,jBegGhost,K).EQ.PBCFLD) then
c.AND.
c     $         VV(I,jBegGhost,K).LT.0.0D0) THEN
c            UU(I,jBegGhost, K) = UU(I,jst,K)
c            AS(I,jst,K) = 2.0D0*TINY
            END IF
  400   CONTINUE
  410 CONTINUE


C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM WALLS
C--------------------------------------------------------------------
      DO 510 J = jst, jend
        DO 500 I = ist, iend
          IF ((PCELL(I,J,kEndGhost).EQ.PBCFLD)) then
c.AND.
c     $        (WW(I,J,kEndGhost+1).GT.0.0D0)) THEN
c             UU(I,J,kEndGhost) = UU(I,J,kend)
c             AT(I,J,kend) = 2.0D0*TINY
          END IF
          IF (PCELL(I,J,kBegGhost).EQ.PBCFLD) then
c.AND.
c     $         WW(I,J,kBegGhost).LT.0.0D0) THEN
c            UU(I,J,kBegGhost) = UU(I,J,kst)
c            AB(I,J,kst) = 2.0D0*TINY
            END IF
  500   CONTINUE
  510 CONTINUE

      RETURN
      END

c*********************************************************************
c
c Revision 1.9  2000/10/07 23:53:03  rawat
c modified bc subroutines
c
c Revision 1.8  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.7  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.6  2000/09/07 23:07:21  rawat
c fixed some bugs in bc and added pressure solver using petsc
c
c Revision 1.5  2000/08/23 06:20:53  bbanerje
c 1) Results now correct for pressure solve.
c 2) Modified BCU, BCV, BCW to add stuff for pressure BC.
c 3) Removed some bugs in BCU, V, W.
c 4) Coefficients for MOM Solve not computed correctly yet.
c
c Revision 1.4  2000/08/17 20:32:02  rawat
c Fixed some bugs
c
c Revision 1.3  2000/08/10 00:56:38  rawat
c added pressure bc for scalar and changed discretization option for velocity
c
c Revision 1.2  2000/07/14 03:45:48  rawat
c completed velocity bc and fixed some bugs
c
c Revision 1.1  2000/07/12 23:59:23  rawat
c added wall bc for u-velocity
c
c*********************************************************************
