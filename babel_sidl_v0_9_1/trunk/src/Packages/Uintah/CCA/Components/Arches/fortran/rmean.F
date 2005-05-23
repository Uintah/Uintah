c
c

      FUNCTION rmean(PROP,I,J,K,IOFF,JOFF,KOFF,
     $     dim_lo, dim_hi)
C-----------------------------------------------------------------------
C     rmean CALCULATES THE MEAN OF THE PROPERTY PROP AT TWO POINTS, 
C     (I,J,K) AND (I-IOFF,J-JOFF,K-KOFF).
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

      integer dim_lo(3)
      integer dim_hi(3)

      double precision prop(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))
      
      double precision rmean

      integer i,j,k,ioff,joff,koff
C-----------------------------------------------------------------------
      RMEAN = PT5*(PROP(I,J,K)+PROP(I-IOFF,J-JOFF,K-KOFF))
      RETURN
      END

c
c Revision 1.1  2000/07/14 05:23:52  bbanerje
c Added scalcoef.F and updated related stuff in C++. scalcoef ==> coefs.f
c in Kumar's code.
c
c
