c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/arradd_fort.h>
c-----------------------------------------------------------------------
c     Written by Seshadri Kumar, October 14, 1996
c     
c     Adds a linear combination of arrays a and b and stores it in c.
c
c     (i.e.,
c
c     c = alpha*a + beta*b, 
c
c     alpha, beta are constants)
c     This subroutine assumes that both a and b are dimensioned
c     (nx,ny,nz).  (ist,jst,kst) and (iend,jend,kend) are the
c     starting and ending locations within which the operation
c     is to be performed
c-----------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST,JST,KST,IEND,JEND,KEND
      integer i,j,k

c*********************************************************************
c     Start :
c*********************************************************************
      IST = indexLow(1)+1
      JST = indexLow(2)+1
      KST = indexLow(3)+1
      IEND = indexHigh(1)-1
      JEND = indexHigh(2)-1
      KEND = indexHigh(3)-1

C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------
      DO 300 K=KST,KEND
         DO 200 J=JST,JEND
            DO 100 I=IST,IEND
               C(I,J,K) = ALPHA*A(I,J,K) + BETA*B(I,J,K)
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:42  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
