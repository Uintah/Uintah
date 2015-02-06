C 
C 
C The MIT License
C 
C Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
C Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
C University of Utah.
C 
C License for the specific language governing rights and limitations under
C Permission is hereby granted, free of charge, to any person obtaining a 
C copy of this software and associated documentation files (the "Software"),
C to deal in the Software without restriction, including without limitation 
C the rights to use, copy, modify, merge, publish, distribute, sublicense, 
C and/or sell copies of the Software, and to permit persons to whom the 
C Software is furnished to do so, subject to the following conditions:
C 
C The above copyright notice and this permission notice shall be included 
C in all copies or substantial portions of the Software.
C 
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
C OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
C THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
C DEALINGS IN THE SOFTWARE.
C 
C
C 
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/mm_explicit_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------
*
*     Written by Seshadri Kumar, March 22, 2002
*
*     This routine solves the linearized first-order explicit equation
*     d/dt(rho*phi) = A*phi + Su
*     Here A is comprised of a 7-point stencil that has east, west, 
*     north, south, top, and bottom neighbors.
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
*	idxLo
*		Starting indices for the calculation (i/p)
*	idxHi
*		Ending indices for calculation (i/p)
C-----------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision vol      
      double precision apo
      double precision rhs
      integer i,j,k

      do 200 k = idxLo(3), idxHi(3)
         do 190 j = idxLo(2), idxHi(2)
            do 180 i = idxLo(1), idxHi(1)

               if (pcell(i,j,k).ne.mmwallid) then

                  vol = sew(i)*sns(j)*stb(k)
                  apo = old_den(i,j,k)*vol/dtime
                  rhs = ae(i,j,k)*old_phi(i+1,j,k)+ 
     $                 aw(i,j,k)*old_phi(i-1,j,k) +
     $                 an(i,j,k)*old_phi(i,j+1,k) + 
     $                 as(i,j,k)*old_phi(i,j-1,k) +
     &                 at(i,j,k)*old_phi(i,j,k+1) + 
     $                 ab(i,j,k)*old_phi(i,j,k-1) +
     &                 su(i,j,k) -ap(i,j,k)*old_phi(i,j,k)
                  phi(i,j,k) = rhs/apo

               else

                  phi(i,j,k) = zero

               endif

 180        continue
 190     continue
 200  continue
      return
      end
