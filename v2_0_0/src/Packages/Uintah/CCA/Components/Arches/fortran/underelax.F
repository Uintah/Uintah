c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/underelax_fort.h>
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
               ap(i,j,k) = ap(i,j,k)/urelax
               su(i,j,k) = su(i,j,k) + (1.0D0 - urelax)*ap(i,j,k)*
     $                                              phi(i,j,k)
 100        continue
 200     continue
 300  continue
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c Revision 1.2  2000/09/29 20:32:42  rawat
c added underrelax to pressure solver
c
c Revision 1.1  2000/08/11 21:26:41  rawat
c added linear solver for pressure eqn
c
c
c*********************************************************************
