c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/mm_explicit_oldvalue_fort.h>
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
c
c     Modification, January 14, 2003
c     This new version sets the value in the mmwallid cell
c     to old_phi rather than zero
c
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

                  phi(i,j,k) = old_phi(i,j,k)

               endif

 180        continue
 190     continue
 200  continue
      return
      end

c*********************************************************************
c
c Revision 1.2  2000/10/09 17:06:30  rawat
c modified momentum solver for multi-patch
c
c Revision 1.1  2000/08/15 00:23:38  rawat
c added explicit solve for momentum and scalar eqns
c
c Revision 1.2  2000/08/11 21:26:41  rawat
c added linear solver for pressure eqn
c
c Revision 1.2  2000/08/01 23:28:46  skumar
c Added residual calculation procedure and modified templates in linear
c solver.  Added template for order-of-magnitude term calculation.
c
c Revision 1.1  2000/04/12 20:31:50  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
