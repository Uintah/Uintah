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
               XP = X(I+1) - XU(I+1)
               TMULT = 0.5D0*(VISCOS(ist,j-1,k)+VISCOS(ist,j,k))/XP

               IF (PCELL(I,J,K).EQ.WALL) THEN
                  if (.not.(yminus.and.j.eq.jst)) then
c                     VV(I,J,K) = 0.0D0
C--------------------------------------------------------------------
C             SOUTHEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
                     SP(I+1,J,K) =SP(I+1,J,K)-TMULT*SNSV(J)*STB(K)
                     AW(I+1,J,K) = 0.0D0
                     cwsav(i+1,j,k) = 0.0D0
                  endif
               else if ((pcell(i,j-1,k).eq.wall).and.
     $                 (pcell(i,j,k).ne.wall)) then
                  SP(I+1,J,K) =SP(I+1,J,K)-TMULT*SNSV(J)*STB(K)
                  AW(I+1,J,K) = 0.0D0
                  cwsav(i+1,j,k) = 0.0D0
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
               IF ((PCELL(iEndGhost,J,K).EQ.PBCFLD
     &              .AND. UU(iEndGhost+1,J,K).GT.0.0D0).or.
     $          (PCELL(iEndGhost, J, K).EQ.outletfld)) THEN
c                  VV(iEndGhost,J,K) = VV(iend,J,K)
c                  AE(iend,J,K) = 2.0D0*TINY
               END IF
               IF (PCELL(iBegGhost,J,K).EQ.PBCFLD
     &              .AND.UU(iBegGhost,J,K).LT.0.0D0) THEN
c                  VV(iBegGhost,J,K) = VV(ist,J,K)
c                  AW(ist,J,K) = 2.0D0*TINY
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
c               VV(I,jEndGhost+1,K) = VV(I,jEndGhost,K)
            END IF
            IF (PCELL(I,jBegGhost,K).EQ.PBCFLD) THEN
c               VV(I,jBegGhost,K) = VV(I,jst,K)
            END IF
 300     CONTINUE
 310  CONTINUE


C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM WALL OUTLET
C--------------------------------------------------------------------
      DO 410 J = jst,jend
         DO 400 I = ist,iend
            IF (PCELL(I,J,kEndGhost).EQ.PBCFLD) then
c     &           .AND.WW(I,J,kEndGhost+1).GT.0.0D0) THEN
c               VV(I,J,kEndGhost) = VV(I,J,kend)
c               AT(I,J,kend) = 2.0D0*TINY
            END IF
            IF (PCELL(I,J,kBegGhost).EQ.PBCFLD) then
c     &           .AND.WW(I,J,kBegGhost).LT.0.0D0) THEN
c               VV(I,J,kBegGhost) = VV(I,J,kst)
c               AB(I,J,kst) = 2.0D0*TINY
            END IF
 400     CONTINUE
 410  CONTINUE
      RETURN
      END
