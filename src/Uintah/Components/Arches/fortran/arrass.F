c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE arrass(indexLow, indexHigh,
     &     A,ALPHA)
C-----------------------------------------------------------------------
C     WRITTEN, OCTOBER 14, 1996, BY SESHADRI KUMAR
C
C     ASSIGNS A CONSTANT ALPHA TO ALL ELEMENTS OF AN ARRAY A, I.E.,
C
C     A = ALPHA,
C
C     (A IS DIMENSIONED (NX,NY,NZ) AS DEFINED IN param1.h) between
C     the limits (IST,JST,KST) AND (IEND,JEND,KEND)
C-----------------------------------------------------------------------
      implicit none

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision A(indexLow(1):indexHigh(1),
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
               A(I,J,K) = ALPHA
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:42  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
