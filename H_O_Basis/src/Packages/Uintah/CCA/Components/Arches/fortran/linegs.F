c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/linegs_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------
*
*     This routine is a relaxation operation on x, given the linear
*     system Ax=b, where A and b are given and the initial guess for
*     x is given.  The matrix A is a 7-diagonal matrix and the values
*     of the diagonals alone are stored, for efficient memory usage.
*     Such a storage scheme obviously requires a special matrix-vector
*     product.  Of the 7 diagonals, the main diagonal and two off-
*     diagonal elements, corresponding to two neighboring elements in 
*     any given direction, are treated implicitly and the other four
*     diagonals are treated implicitly with the source term, leading
*     to a tridiagonal system.  If LSWPWE is true, then the linear
*     system is treated implicitly in the x-direction, i.e., AE and
*     AW are treated implicitly.  If LSWPSN is true, then the linear
*     system is treated implicitly in the y-direction, and if LSWPBT
*     is true, then the linear system is treated implicitly in the 
*     z-direction.  If all three logicals are true, this preconditioning
*     operation corresponds to one iteration of the ADI (Alternating-
*     Direction-Implicit) technique
*
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
      logical*1 lswpwe
      logical*1 lswpsn
      logical*1 lswpbt
      

      integer i,j,k, ii, jj, kk, nkused

      double precision convwe
      double precision convsn
      double precision convbt
      double precision rhs
      double precision apc
      double precision denom
      lswpwe = .TRUE.
      lswpsn = .TRUE.
      lswpbt = .TRUE.
c-----------------------------------------------------------------------
c     west to east x direction sweep  
c-----------------------------------------------------------------------
      convwe = zero
      if (lswpwe) then
c-----------------------------------------------------------------------
c     commence s-n and b-t traverse
c-----------------------------------------------------------------------
        do 200 k = idxLo(3), idxHi(3)
          do 190 j = idxLo(2), idxHi(2)

            e1(idxLo(1) - 1) = zero
            f1(idxLo(1) - 1) = phi(idxLo(1)-1, j, k)
c----------------------------------------------------------------------- 
c     commence w-e sweep 
c-----------------------------------------------------------------------
            do 140 i = idxLo(1), idxHi(1)
c----------------------------------------------------------------------- 
c     assemble tdma coefficients
c-----------------------------------------------------------------------
              rhs = an(i,j,k)*phi(i,j+1,k) + as(i,j,k)*phi(i,j-1,k)
     &             +at(i,j,k)*phi(i,j,k+1) + ab(i,j,k)*phi(i,j,k-1)
     &             +su(i,j,k)
              apc = theta*(an(i,j,k) + as(i,j,k) + at(i,j,k))
              rhs = rhs - apc*phi(i,j,k)
c-----------------------------------------------------------------------
c     calculate coefficients of recurrence formula
c-----------------------------------------------------------------------
              denom = one/(ap(i,j,k) - apc - aw(i,j,k)*e1(i-1))
              e1(i)  = ae(i,j,k)*denom
              f1(i)  = (rhs + aw(i,j,k)*f1(i-1))*denom
  140       continue
c-----------------------------------------------------------------------
c     obtain new phi's
c-----------------------------------------------------------------------
            do 180 ii = idxLo(1), idxHi(1)
              i = idxHi(1) + idxLo(1) -ii

              phi(i,j,k) = (e1(i)*phi(i+1,j,k)) + f1(i)
              convwe = convwe + min(abs(e1(i)*phi(i+1,j,k)),
     &                 abs(f1(i)))/max(tiny,
     &                 abs(e1(i)*phi(i+1,j,k)),abs(f1(i)))

  180       continue
  190     continue
  200   continue
      end if

c-----------------------------------------------------------------------
c       south to north r or y direction sweep   
c-----------------------------------------------------------------------
      convsn = zero
      if (lswpsn) then
c-----------------------------------------------------------------------
c     commence w-e and b-t traverse
c-----------------------------------------------------------------------
        do 400 k = idxLo(3), idxHi(3)
          do 390 i = idxLo(1), idxHi(1)

            e2(idxLo(2) - 1 ) = zero
            f2(idxLo(2) - 1 ) = phi(i, idxLo(2) - 1 , k)
c-----------------------------------------------------------------------
c     commence s-n sweep
c-----------------------------------------------------------------------
            do 340 j = idxLo(2), idxHi(2)
c-----------------------------------------------------------------------
c     assemble tdma coefficients
c-----------------------------------------------------------------------
              rhs = ae(i,j,k)*phi(i+1,j,k) + aw(i,j,k)*phi(i-1,j,k)
     &             +at(i,j,k)*phi(i,j,k+1) + ab(i,j,k)*phi(i,j,k-1)
     &             +su(i,j,k)
              apc = theta*(ae(i,j,k) + aw(i,j,k) + at(i,j,k))
              rhs = rhs - apc*phi(i,j,k)
c-----------------------------------------------------------------------
c     calculate coefficients of recurrence formula
c-----------------------------------------------------------------------
              denom = one/(ap(i,j,k) - apc - as(i,j,k)*e2(j-1))
              e2(j)  = an(i,j,k)*denom
              f2(j)  = (rhs + as(i,j,k)*f2(j-1))*denom              
  340       continue
c-----------------------------------------------------------------------
c     obtain new phi's
c-----------------------------------------------------------------------
            do 380 jj = idxLo(2), idxHi(2)
              j = idxHi(2) + idxLo(2) -jj

              phi(i,j,k) = e2(j)*phi(i,j+1,k) + f2(j)
              convsn = convsn + min(abs(e2(j)*phi(i,j+1,k)),
     &                 abs(f2(j)))/max(abs(e2(j)*phi(i,j+1,k)),
     &                 abs(f2(j)),tiny)

  380       continue
  390     continue
  400   continue
      end if 

c-----------------------------------------------------------------------
c     bottom to top z or theta direction sweep  
c-----------------------------------------------------------------------
      convbt = zero
      if (lswpbt) then
c-----------------------------------------------------------------------
c     commence w-e and s-n traverse
c-----------------------------------------------------------------------
        do 600 j = idxLo(2), idxHi(2)
          do 590 i = idxLo(1), idxHi(1)
            e3(idxLo(3) - 1) = zero
            f3(idxLo(3) - 1) = phi(i, j, idxLo(3) - 1)
c-----------------------------------------------------------------------
c     commence b-t sweep
c-----------------------------------------------------------------------
            do 540 k = idxLo(3), idxHi(3)
c-----------------------------------------------------------------------
c     assemble tdma coefficients
c-----------------------------------------------------------------------
              rhs = ae(i,j,k)*phi(i+1,j,k) + aw(i,j,k)*phi(i-1,j,k)
     &             +an(i,j,k)*phi(i,j+1,k) + as(i,j,k)*phi(i,j-1,k)
     &             +su(i,j,k)
              apc = theta*(ae(i,j,k) + aw(i,j,k) + an(i,j,k))
              rhs = rhs - apc*phi(i,j,k)
c-----------------------------------------------------------------------
c     calculate coefficients of recurrence formula
c-----------------------------------------------------------------------
              denom = one/(ap(i,j,k) - apc - ab(i,j,k)*e3(k-1))
              e3(k)  = at(i,j,k)*denom
              f3(k)  = (rhs + ab(i,j,k)*f3(k-1))*denom
  540       continue
c-----------------------------------------------------------------------
c     obtain new phi's
c-----------------------------------------------------------------------
            do 580 kk = idxLo(3), idxHi(3)
              k = idxHi(3) + idxLo(3) - kk

              phi(i,j,k) = e3(k)*phi(i,j,k+1) + f3(k)
              convbt = convbt + min(abs(e3(k)*phi(i,j,k+1)),
     &                 abs(f3(k)))/max(abs(e3(k)*phi(i,j,k+1)),
     &                 abs(f3(k)),tiny)

  580       continue
  590     continue
  600   continue
      end if
c
      return
      end

c*********************************************************************
c
c Revision 1.3  2000/08/15 00:23:38  rawat
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
