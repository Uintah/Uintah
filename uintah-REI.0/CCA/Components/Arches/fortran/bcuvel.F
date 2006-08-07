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
      integer i, j, k


c*******************u*************************************************
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
         DO 120 K = kst, kend
            DO 110 J = jst, jend
               I = ist -1
               if (pcell(I,J,K).eq.wall) then
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

      RETURN
      END

