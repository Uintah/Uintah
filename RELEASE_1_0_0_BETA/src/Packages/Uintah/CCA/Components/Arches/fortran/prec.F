c*********************************************************************
c
c
c*********************************************************************

      subroutine prec(eps,digmax)

      implicit none

#include "param4.h"     

      double precision
     $     eps,digmax
      double precision
     $     s
C
C     Purpose: To calculate machine precision and maximum digits
C     of inaccuracy possible on the platform being used.
C     Author: Seshadri Kumar
C     Date: June 2, 1997
C      
      eps = one
   10 eps = eps/two
      s = one + eps
      if (s .gt. one) go to 10
      eps = two*eps
      digmax = -log10(abs(eps))
      return
      end

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************

