c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_vel_fort.h>
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

      do 200 k = idxLo(3), idxHi(3)
         do 190 j = idxLo(2), idxHi(2)
            do 180 i = idxLo(1), idxHi(1)
               vol = sew(i)*sns(j)*stb(k)
            apo = 0.5D0*(old_den(i,j,k)+old_den(i-ioff,j-joff,k-koff))
     $                *vol/dtime
               rhs = ae(i,j,k)*old_phi(i+1,j,k)+ 
     $              aw(i,j,k)*old_phi(i-1,j,k) +
     $              an(i,j,k)*old_phi(i,j+1,k) + 
     $              as(i,j,k)*old_phi(i,j-1,k) +
     &              at(i,j,k)*old_phi(i,j,k+1) + 
     $              ab(i,j,k)*old_phi(i,j,k-1) +
     &              su(i,j,k) -ap(i,j,k)*old_phi(i,j,k)
               phi(i,j,k) = rhs/apo
 180        continue
 190     continue
 200  continue
      return
      end


