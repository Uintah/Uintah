c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/bcwvel_fort.h>
#include "param4.h"
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision XP, TMULT

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c------------------------------------------------------------------ 
c     interior intrusions
c------------------------------------------------------------------ 
      do 820 k=kst, kend
         do 810 j=jst, jend
            do 800 i=ist, iend

               if (pcell(i,j,k).eq.wall.or.pcell(i,j,k-1).eq.wall) then
                call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     &                          as(i,j,k),at(i,j,k),ab(i,j,k),
     &                          su(i,j,k),sp(i,j,k),one)
                call fixval(zero,cesav(i,j,k),cwsav(i,j,k),cnsav(i,j,k),
     &                 cssav(i,j,k),ctsav(i,j,k),cbsav(i,j,k),su(i,j,k),
     &                 sp(i,j,k),one)
               end if

 800        continue
 810     continue
 820  continue


c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      if (xminus) then
         I = ist - 1
         DO 120 K = kst, kend
            DO 110 J = jst, jend
C--------------------------------------------------------------------
C             BOTTOMEAST FACE OF INTRUSION\WALL
C--------------------------------------------------------------------
               IF (((PCELL(I,J,K).EQ.WALL).or.
     &              (pcell(i,j,k-1).eq.wall)).and.
     &             (pcell(i+1,j,k).ne.wall).and.
     &             (pcell(i+1,j,k-1).ne.wall)) then
                  XP = X(I+1) - XU(I+1)
                  TMULT = 0.5D0*(VISCOS(I+1,j,k-1)+VISCOS(I+1,j,k))/XP
                  SP(I+1,J,K) = SP(I+1,J,K)-TMULT*SNS(J)*STBW(K)
                  AW(I+1,J,K) = 0.0D0
                  cwsav(i+1,j,k) = 0.0D0
                END IF
 110        CONTINUE
 120     CONTINUE
      endif

      RETURN
      END
