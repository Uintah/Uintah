C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
C 
#include <CCA/Components/Arches/fortran/mascal_scalar_fort.h>
#include "param4.h"
C--------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
*     last modified, September 15, 1998
C
C     THIS SUBROUTINE CALCULATES THE MASS SOURCE TERM DUE TO 
C     CONTINUITY AND THE UTILIZATION OF THE CONSERVATIVE FORM IN THE 
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
      logical*1 l2up, leno

      integer i
      integer j
      integer k

      l2up = .false.
      leno = .false.
      if (conv_scheme.eq.0) then
        l2up = .true.
      else
        leno = .true.
      endif

      do 420 k = valid_lo(3), valid_hi(3)
         do 410 j = valid_lo(2), valid_hi(2) 
            do 400 i = valid_lo(1), valid_hi(1)

c
c          note: with d's in the a's, a's will always be finite
c                unless explicitly set to zero
c                and at symtry planes the velocity is also zero
c
           if (l2up) then
               if (ae(i,j,k).lt.0.0D0) cesav(i,j,k) = zero
               if (aw(i,j,k).lt.0.0D0) cwsav(i,j,k) = zero
               if (an(i,j,k).lt.0.0D0) cnsav(i,j,k) = zero
               if (as(i,j,k).lt.0.0D0) cssav(i,j,k) = zero
               if (at(i,j,k).lt.0.0D0) ctsav(i,j,k) = zero
               if (ab(i,j,k).lt.0.0D0) cbsav(i,j,k) = zero
           else
               if (abs(ae(i,j,k)).lt.tiny) cesav(i,j,k) = zero
               if (abs(aw(i,j,k)).lt.tiny) cwsav(i,j,k) = zero
               if (abs(an(i,j,k)).lt.tiny) cnsav(i,j,k) = zero
               if (abs(as(i,j,k)).lt.tiny) cssav(i,j,k) = zero
               if (abs(at(i,j,k)).lt.tiny) ctsav(i,j,k) = zero
               if (abs(ab(i,j,k)).lt.tiny) cbsav(i,j,k) = zero
           endif


 400        continue
 410     continue
 420  continue

c-----calculate mass source------------------------------------------

      do 520 k = valid_lo(3), valid_hi(3)
         do 510 j = valid_lo(2), valid_hi(2)
            do 500 i = valid_lo(1), valid_hi(1)
 
               smp = cnsav(i,j,k) - cssav(i,j,k) + cesav(i,j,k) -
     &              cwsav(i,j,k) + ctsav(i,j,k) - cbsav(i,j,k) 

               if (leno) smp = zero

               su(i,j,k) = su(i,j,k) - smp*phi(i,j,k)

 500        continue
 510     continue
 520  continue

c--------------------------------------------------------------------
      return
      end
