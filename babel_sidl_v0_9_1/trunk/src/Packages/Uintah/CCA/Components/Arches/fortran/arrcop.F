c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/addcop_fort.h>
#else
      SUBROUTINE arrcop(indexLow, indexHigh,
     &     A,B,ALPHA)
C-----------------------------------------------------------------------
C     WRITTEN, OCTOBER 14, 1996, BY SESHADRI KUMAR 
C
C     COPIES ARRAY B TO ARRAY A WITH A SCALAR MULTIPLE, I.E.,
C
C     A = ALPHA*B
C
C     THIS SUBROUTINE ASSUMES THAT BOTH A AND B ARE DIMENSIONED
C     (NX,NY,NZ).  (IST,JST,KST) AND (IEND,JEND,KEND) ARE THE
C     STARTING AND ENDING LOCATIONS WITHIN WHICH THE COPYING IS
C     TO BE DONE
C-----------------------------------------------------------------------
      implicit none

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision A(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision B(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision ALPHA

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
               A(I,J,K) = ALPHA*B(I,J,K)
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
