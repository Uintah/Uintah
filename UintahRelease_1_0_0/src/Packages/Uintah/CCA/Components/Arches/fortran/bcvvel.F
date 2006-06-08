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
         DO 120 K = kst, kend
            DO 110 J = jst, jend
               I = ist - 1
               XP = X(I+1) - XU(I+1)
               TMULT = 0.5D0*(VISCOS(ist,j-1,k)+VISCOS(ist,j,k))/XP

               IF (PCELL(I,J,K).EQ.WALL) THEN
                  if (.not.(yminus.and.j.eq.jst)) then
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

      RETURN
      END
