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
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/mm_explicit_vel_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------
*
*     This routine is a relaxation operation on x, given the linear
*     system Ax=b, where A and b are given and the initial guess for
*     x is given. 
*     Passed Variables:
*
*	PHI	Variable for which equation is being solved (i/p) 
*               (o/p)(=x in the previous discussion).  PHI on input
*               is the initial guess for PHI, and is overwritten by its
*               new value on exit.
*	AE,AW,AN,AS,AT,AB,AP
*		Coefficients of the seven-diagonal finite difference
*		equation (i/p) (=A in the previous discussion).
*	SU	Source term for the PHI-equation (i/p) (=b in the previous
*               discussion).
*	THETA 	Stone implicit acceleration factor (i/p). (zero to one)
*	LSWPWE,LSWPSN,LSWPBT
*		Logicals to determine whether sweeps in linegs should
*		be done in the x-, y-, and z-directions respectively
*		(i/p).
*     
*	IST,JST,KST
*		Starting indices for the calculation (i/p)
*	IEND,JEND,KEND
*		Ending indices for calculation (i/p)
C-----------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision vol      
      double precision apo
      double precision rhs
      integer i,j,k
      integer idir

      do 200 k = idxLo(3), idxHi(3)
         do 190 j = idxLo(2), idxHi(2)
            do 180 i = idxLo(1), idxHi(1)
               
               if ((pcell(i,j,k).ne.mmwallid) .and.
     $              (pcell(i-ioff, j-joff, k-koff)
     $              .ne.mmwallid)) then
               
                  vol = sew(i)*sns(j)*stb(k)
                  apo = 0.5*(old_den(i,j,k)+
     $                 old_den(i-ioff,j-joff,k-koff))
     $                 *vol/dtime
                  rhs = ae(i,j,k)*old_phi(i+1,j,k)+ 
     $                 aw(i,j,k)*old_phi(i-1,j,k) +
     $                 an(i,j,k)*old_phi(i,j+1,k) + 
     $                 as(i,j,k)*old_phi(i,j-1,k) +
     &                 at(i,j,k)*old_phi(i,j,k+1) + 
     $                 ab(i,j,k)*old_phi(i,j,k-1) +
     &                 su(i,j,k) -ap(i,j,k)*old_phi(i,j,k)
                  phi(i,j,k) = rhs/apo

#if 0
                  if ( (koff.eq.1) .and.
     $                 (i.eq.3).and.(j.eq.9).and.(k.eq.9)) then
                     print*, 'solution at 3,9,9 = ',phi(i,j,k)
                     print*, 'apo = ',apo
                     print*, 'rhs = ',rhs
                     print*, 'ae = ',ae(i,j,k)
                     print*, 'old_phi_e = ',old_phi(i+1,j,k)
                     print*, 'aw = ',aw(i,j,k)
                     print*, 'old_phi_w = ',old_phi(i-1,j,k)
                     print*, 'an = ',an(i,j,k)
                     print*, 'old_phi_n = ',old_phi(i,j+1,k)
                     print*, 'as = ',as(i,j,k)
                     print*, 'old_phi_s = ',old_phi(i,j-1,k)
                     print*, 'at = ',at(i,j,k)
                     print*, 'old_phi_t = ',old_phi(i,j,k+1)
                     print*, 'ab = ',ab(i,j,k)
                     print*, 'old_phi_b = ',old_phi(i,j,k-1)

                     print*, 'source term = ',su(i,j,k)
                     print*, 'diagonal = ',ap(i,j,k)
                     print*, 'old_phi_p = ',old_phi(i,j,k)

                     print*, 'old density at P = ',old_den(i,j,k)
                     print*, 'old density at B = ',old_den(i,j,k-1)
                  endif
#endif 

               else

                  phi(i,j,k) = zero
c     may need to change this if mm block is allowed to move?

               endif

 180        continue
 190     continue
 200  continue

      idir = ioff*1 + joff*2 + koff*3

      return
      end
