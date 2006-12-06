c
c $Id $
c 

#include <Packages/Uintah/CCA/Components/Arches/fortran/mascal_fort.h>
#include "param4.h"
C--------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
*     last modified, September 15, 1998
C
C     THIS SUBROUTINE CALCULATES THE MASS SOURCE TERM DUE TO 
C     CONTINUITY AND THE USE OF THE CONSERVATIVE FORM IN THE 
C     PARTIAL DIFFERENTIAL EQUATION
C
C     PASSED VARIABLES:
C
C	PHI	Variable for which equation is being solved (i/p)
C	AE,AW,AN,AS,AT,AB,AP
C		Coefficients of the seven-diagonal finite difference
C		equation (o/p)
C	SU	Nonlinear source term, array (i/p and o/p)
C	valid_lo
C		Starting indices for the calculation (i/p)
C	valid_hi
C		Ending indices for calculation (i/p)
C	CESAV,CWSAV,CNSAV,CSSAV,CTSAV,CBSAV
C		Convection fluxes on the east,west,north,south,top,
C		and bottom faces of the P-cell (i/p)
C
C--------------------------------------------------------------------

*
*     declare local variables
*
      double precision smp
      logical*1 l2up

      integer i
      integer j
      integer k

      l2up = .false.

      do 420 k = valid_lo(3), valid_hi(3)
         do 410 j = valid_lo(2), valid_hi(2) 
            do 400 i = valid_lo(1), valid_hi(1)

c
c          note: with d's in the a's, a's will always be finite
c                unless explicitly set to zero
c                and at symtry planes the velocity is also zero
c
               if (abs(ae(i,j,k)).lt.tiny) cesav(i,j,k) = zero
               if (abs(aw(i,j,k)).lt.tiny) cwsav(i,j,k) = zero
               if (abs(an(i,j,k)).lt.tiny) cnsav(i,j,k) = zero
               if (abs(as(i,j,k)).lt.tiny) cssav(i,j,k) = zero
               if (abs(at(i,j,k)).lt.tiny) ctsav(i,j,k) = zero
               if (abs(ab(i,j,k)).lt.tiny) cbsav(i,j,k) = zero

 400        continue
 410     continue
 420  continue

c-----calculate mass source------------------------------------------

      do 520 k = valid_lo(3), valid_hi(3)
         do 510 j = valid_lo(2), valid_hi(2)
            do 500 i = valid_lo(1), valid_hi(1)
 
               smp = cnsav(i,j,k) - cssav(i,j,k) + cesav(i,j,k) -
     &              cwsav(i,j,k) + ctsav(i,j,k) - cbsav(i,j,k) 

               su(i,j,k) = su(i,j,k) - smp*phi(i,j,k)

 500        continue
 510     continue
 520  continue

c--------------------------------------------------------------------
      return
      end

c
c $Log $
c 
