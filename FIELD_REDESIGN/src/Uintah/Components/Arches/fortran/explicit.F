c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE explicit(domLoU, domHiU, idxLo, idxHi,
     &     PHI, old_PHI, 
     &     AE,AW,AN,AS,AT,AB,AP,SU,
     $     domLo, domHi, old_den, sew, sns, stb, dtime)
c     &     lswpwe, lswpsn, lswpbt)
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
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3)
      integer domHiU(3)
      integer idxLo(3)
      integer idxHi(3)
      integer domLo(3)
      integer domHi(3)
      double precision PHI(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision old_PHI(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision AP(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision AE(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision AW(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision AN(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision AS(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision AT(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision AB(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision SU(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision old_Den(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision sew(domLo(1):domHi(1))
      double precision sns(domLo(2):domHi(2))
      double precision stb(domLo(3):domHi(3))
      double precision dtime

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
               vol = sew(i)*sns(j)*stb(k)
               apo = old_den(i,j,k)*vol/dtime
               rhs = ae(i,j,k)*old_phi(i+1,j,k)+ 
     $              aw(i,j,k)*old_phi(i-1,j,k) +
     $              an(i,j,k)*old_phi(i,j+1,k) + 
     $              as(i,j,k)*old_phi(i,j-1,k) +
     &              at(i,j,k)*old_phi(i,j,k+1) + 
     $              ab(i,j,k)*old_phi(i,j,k-1) +
     &              su(i,j,k) -ap(i,j,k)*old_phi(i,j,k)
               phi(i,j,k) = rhs/apo
 180        continue
 190        continue
 200        continue
      return
      end

c*********************************************************************
c
c $Log$
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
