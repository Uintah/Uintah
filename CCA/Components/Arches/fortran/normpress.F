c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE normpress(domLo, domHi, idxLo, idxHi,
     &     PRESS,
     &     refPress)
C-----------------------------------------------------------------------
C     EXPLANATION OF SYMBOLS USED:
C
C	AP	Diagonal coefficient (i/p)
C	IST,JST,KST
C		Starting indices for the calculation (i/p)
C	IEND,JEND,KEND
C		Ending indices for calculation (i/p)
C	PHI	Variable for which equation is being solved
c       urelax  underrelaxation parameter for the PHI-equation
C	SU	Source term for the PHI equation
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLo(3)
      integer domHi(3)
      integer idxLo(3)
      integer idxHi(3)
      double precision PRESS(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision refPress

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i,j,k

C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------

      do 300 k = idxLo(3), idxHi(3)
         do 200 j = idxLo(2), idxHi(2)
            do 100 i = idxLo(1), idxHi(1)
               press(i,j,k) = press(i,j,k) - refPress
 100        continue
 200     continue
 300  continue
#if 0
      print*, 'value of pressure after normalization = ',
     $     'at wall location 5, 9, 9 = ',press(5,9,9)

      print*, 'value of pressure after normalization = ',
     $     'at wall location 5, 9, 10 = ',press(5,9,10)

      print*, 'value of pressure after normalization = ',
     $     'at wall location 5, 10, 9 = ',press(5,10,9)

      print*, 'value of pressure after normalization = ',
     $     'at wall location 5, 10, 10 = ',press(5,10,10)
#endif
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/08/11 21:26:41  rawat
c added linear solver for pressure eqn
c
c
c*********************************************************************
