c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/arrl1_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
C
C     CALCULATES THE L1 VECTOR NORM OF THE MATRIX A AND STORES IT IN
C     ALPHA
C
C     I.E., ALPHA = SUMMATION(ABS(AA(I,J,K))
C
C     ALSO CALCULATES THE L1 MEAN VALUE OF AA, ALPAVE.
C
C-----------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************

      integer i,j,k,NTOTAL

c*********************************************************************
c     Start :
c*********************************************************************

C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------
      ALPHA = ZERO
      DO 300 K = valid_lo(3), valid_hi(3)
         DO 200 J = valid_lo(2), valid_hi(2)
            DO 100 I = valid_lo(1), valid_hi(1)
               ALPHA = ALPHA + ABS(AA(I,J,K))
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE
      NTOTAL = 
     $     (valid_hi(3) - valid_lo(3) + 1)*
     $     (valid_hi(2) - valid_lo(2) + 1)*
     $     (valid_hi(1) - valid_lo(1) + 1)
      ALPAVE = ALPHA/NTOTAL
      RETURN
      END

c*********************************************************************
c
c Revision 1.2  2000/08/01 23:28:46  skumar
c Added residual calculation procedure and modified templates in linear
c solver.  Added template for order-of-magnitude term calculation.
c
c Revision 1.1  2000/04/12 20:31:42  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

