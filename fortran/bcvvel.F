c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/bcvvel_fort.h>
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
      integer idxUInlet, idxWInlet
      integer i, j, k


      double precision XP, ZP, TMULT

c*********************************************************************
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

               if (pcell(i,j,k).eq.wall.or. pcell(i,j-1,k).eq.wall)
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
               I = ist - 1
               IF (PCELL(I,J,K).EQ.WALL) THEN
                  VV(I,J,K) = 0.0
C--------------------------------------------------------------------
C             SOUTHEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                  IF (PCELL(I+1,J,K).NE.WALL.AND.
     &                 PCELL(I+1,J-1,K).NE.WALL.AND.
     $                 .NOT.(yminus.and.j.eq.jst))THEN
                     XP = X(I+1) - XU(I+1)
                     TMULT = VISCOS/XP
                     SP(I+1,J,K) =SP(I+1,J,K)-TMULT*SNSV(J)*STB(K)
                     AW(I+1,J,K) = 0.0
                  END IF
C--------------------------------------------------------------------
C             NORTHEAST FACE OF INTRUSION/WALL
C--------------------------------------------------------------------
                  IF (J.NE.JEND) THEN
                     IF (PCELL(I,J+1,K).NE.WALL) THEN
                        IF (PCELL(I+1,J+1,K).NE.WALL.AND.
     &                       PCELL(I+1,J,K).NE.WALL) THEN
                           XP = X(I+1)-XU(I+1)
                           TMULT = VISCOS/XP
                           SP(I+1,J+1,K) = SP(I+1,J+1,K)-TMULT*
     $                          SNSV(J+1)*STB(K)
                           AW(I+1,J+1,K) = 0.0
                        END IF
                     END IF
                  endif
C--------------------------------------------------------------------
C             SOUTHEAST FACE OF INTRUSION/WALL, only if next to south patch
C--------------------------------------------------------------------
                  IF ((J.EQ.JST+1).AND.(.NOT.yminus)) THEN
                     IF (PCELL(I,J-1,K).NE.WALL) THEN
                        IF (PCELL(I+1,J-1,K).NE.WALL.AND.
     &                       PCELL(I+1,J,K).NE.WALL) THEN
                           XP = X(I+1)-XU(I+1)
                           TMULT = VISCOS/XP
                           SP(I+1,J-1,K) = SP(I+1,J-1,K)-TMULT*
     $                          SNSV(J-1)*STB(K)
                           AW(I+1,J-1,K) = 0.0
                        END IF
                     END IF
                  endif
               endif
 110        CONTINUE
 120     CONTINUE
      endif

C--------------------------------------------------------------------
C     PRESSURE BOUNDARY CONDITONS
C       EAST AND/OR WEST WALLS
C--------------------------------------------------------------------
      if (xminus.or.xplus) then
         DO 210 K = kst,kend
            DO 200 J = jst,jend
               IF (PCELL(iEndGhost,J,K).EQ.PBCFLD
     &              .AND. UU(iEndGhost+1,J,K).GT.0.0) THEN
                  VV(iEndGhost,J,K) = VV(iend,J,K)
                  AE(iend,J,K) = 2.0*TINY
               END IF
               IF (PCELL(iBegGhost,J,K).EQ.PBCFLD
     &              .AND.UU(iBegGhost,J,K).LT.0.0) THEN
                  VV(iBegGhost,J,K) = VV(ist,J,K)
                  AW(ist,J,K) = 2.0*TINY
               END IF
 200        CONTINUE
 210     CONTINUE
      endif

C--------------------------------------------------------------------
C     PRESSURE BOUNDARY CONDITONS
C       North AND/OR South WALLS
C--------------------------------------------------------------------
      DO 310 K = kst,kend
         DO 300 I = ist,iend
            IF (PCELL(I,jEndGhost,K).EQ.PBCFLD) THEN
               VV(I,jEndGhost+1,K) = VV(I,jEndGhost,K)
            END IF
            IF (PCELL(I,jBegGhost,K).EQ.PBCFLD) THEN
               VV(I,jBegGhost,K) = VV(I,jst,K)
            END IF
 300     CONTINUE
 310  CONTINUE


C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM WALL OUTLET
C--------------------------------------------------------------------
      DO 410 J = jst,jend
         DO 400 I = ist,iend
            IF (PCELL(I,J,kEndGhost).EQ.PBCFLD
     &           .AND.WW(I,J,kEndGhost+1).GT.0.0) THEN
               VV(I,J,kEndGhost) = VV(I,J,kend)
               AT(I,J,kend) = 2.0*TINY
            END IF
            IF (PCELL(I,J,kBegGhost).EQ.PBCFLD
     &           .AND.WW(I,J,kBegGhost).LT.0.0) THEN
               VV(I,J,kBegGhost) = VV(I,J,kst)
               AB(I,J,kst) = 2.0*TINY
            END IF
 400     CONTINUE
 410  CONTINUE
      RETURN
      END

c*********************************************************************
c
c Revision 1.9  2000/10/09 17:06:30  rawat
c modified momentum solver for multi-patch
c
c Revision 1.8  2000/10/08 18:56:41  rawat
c fixed the solver for multi
c
c Revision 1.7  2000/10/07 23:53:03  rawat
c modified bc subroutines
c
c Revision 1.6  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.5  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.4  2000/09/07 23:07:21  rawat
c fixed some bugs in bc and added pressure solver using petsc
c
c Revision 1.3  2000/08/23 06:20:53  bbanerje
c 1) Results now correct for pressure solve.
c 2) Modified BCU, BCV, BCW to add stuff for pressure BC.
c 3) Removed some bugs in BCU, V, W.
c 4) Coefficients for MOM Solve not computed correctly yet.
c
c Revision 1.2  2000/08/17 20:32:02  rawat
c Fixed some bugs
c
c Revision 1.1  2000/07/14 03:58:43  bbanerje
c Added missed files.
c
c Revision 1.1  2000/07/12 23:59:23  rawat
c added wall bc for u-velocity
c
c*********************************************************************
