c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE explicit(domLoU, domHiU,domLoUng, domHiUng,
     $     idxLo, idxHi,
     &     PHI, old_PHI, 
     &     AE,AW,AN,AS,AT,AB,AP,SU,
     $     domLo, domHi,domLowg, domHiwg, old_den, sew, 
     $     sns, stb, dtime)
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
      integer domLoUng(3)
      integer domHiUng(3)
      integer idxLo(3)
      integer idxHi(3)
      integer domLo(3)
      integer domHi(3)
      integer domLowg(3)
      integer domHiwg(3)
      double precision PHI(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision old_PHI(domLoU(1):domHiU(1), 
     &     domLoU(2):domHiU(2), domLoU(3):domHiU(3))
      double precision AP(domLoUng(1):domHiUng(1), 
     &     domLoUng(2):domHiUng(2), domLoUng(3):domHiUng(3))
      double precision AE(domLoUng(1):domHiUng(1), 
     &     domLoUng(2):domHiUng(2), domLoUng(3):domHiUng(3))
      double precision AW(domLoUng(1):domHiUng(1), 
     &     domLoUng(2):domHiUng(2), domLoUng(3):domHiUng(3))
      double precision AN(domLoUng(1):domHiUng(1), 
     &     domLoUng(2):domHiUng(2), domLoUng(3):domHiUng(3))
      double precision AS(domLoUng(1):domHiUng(1), 
     &     domLoUng(2):domHiUng(2), domLoUng(3):domHiUng(3))
      double precision AT(domLoUng(1):domHiUng(1), 
     &     domLoUng(2):domHiUng(2), domLoUng(3):domHiUng(3))
      double precision AB(domLoUng(1):domHiUng(1), 
     &     domLoUng(2):domHiUng(2), domLoUng(3):domHiUng(3))
      double precision SU(domLoUng(1):domHiUng(1), 
     &     domLoUng(2):domHiUng(2), domLoUng(3):domHiUng(3))
      double precision old_Den(domLo(1):domHi(1), 
     &     domLo(2):domHi(2), domLo(3):domHi(3))
      double precision sew(domLowg(1):domHiwg(1))
      double precision sns(domLowg(2):domHiwg(2))
      double precision stb(domLowg(3):domHiwg(3))
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
