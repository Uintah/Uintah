c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/solve_fort.h>
#else
c      SUBROUTINE solve(indexLow, indexHigh,
     &     FF,FCOR,
     &     AE,AW,AN,AS,AT,AB,AP,SU,
     $     RESF,
     &     GAMF,URFF,TSOLV,THETA,CONVL,R_NORM,
     $     NI,NJ,NK,
     &     ILCNT,IEQN,ISPF,NSWPF,ISOPT,
     $     LSWPWE,LSWPSN,LSWPBT,LPARL,LTEST,LLEFT,
     &     tmpXYZArr1, tmpXYZArr2,
     &     tmpXArr1, tmpXArr2,
     $     NEQ, CLABEL)
C-----------------------------------------------------------------------
*     WRITTEN BY: SESHADRI KUMAR, OCTOBER 14, 1996
*     last modified: Seshadri Kumar, August 13, 1998
*
*     SOLVE solves the linearized PDE (finite-difference equation)
*     for any variable F and yields the correction FCOR to F,
*     given the diagonals of its seven-point stencil difference 
*     equation, AE,AW,AN,AS,AT,AB, and AP, and the right-hand 
*     side source term, SU.  If the Jacobi option, LPARL, is 
*     true, then SOLVE returns only the correction. However, 
*     if LPARL is false, then SOLVE returns the updated
*     value of F by adding FCOR to F.
*
*     Passed variables:
*	F	variable to be solved for.  On input, F is the initial
*		guess.  On output, if LPARL is true, F is also the new 
*               solution (i/p) (o/p)
*	FCOR	correction to F (o/p)
*	AE,AW,AN,AS,AT,AB,AP
*		Coefficients of the seven-diagonal finite difference
*		equation (i/p) (=A in Ax=b)
*	SU	Source term for the F-equation (i/p) (=b in Ax=b)
*	RESF	Array containing the residual of the F-equation (o/p)
*	GAMF	linear equation convergence tolerance for the
*		F-equation (i/p)
*	URFF	Under-relaxation factor for F-equation (i/p)
*	TSOLV	CPU time taken for equation solution (i/p and o/p)
*	THETA 	Stone implicit acceleration factor (i/p)
*	CONVL	round-off error in linear equation solver (o/p)
*	R_NORM	normalization factor for residual to determine
*		its closeness to zero in relative terms (i/p)
*		This is used if a user-specified normalization factor
*		is used for this evaluation instead of using the residual
*		at the start for the evaluation
*	IST,JST,KST,IEND,JEND,KEND
*		Locations of the starting and ending indices for the
*		interior points of the F-equation, the points that need
*		to be solved for (i/p)
*	NI	Number of total points (including ghost points) in
*		the I-direction (X-direction) in the domain (i/p)
*	NJ	Number of total points (including ghost points) in
*		the J-direction (Y- or R-direction)
*	NK	Number of total points (including ghost points) in
*		the K-direction (Z- or THETA-direction)
*	ILCNT	Total number of calls to subroutine linegs in the
*		entire calculation (i/p and o/p)
*	NITER	Count variable for number of calls made to subroutine
*		fixpt (i/p)
*	IEQN	Equation number of the F-equation (i/p) (see setup_gmres.f)
*	ISPF	Number of calls made to linegs in this call to pdef
*		(o/p)
*	NSWPF	Total number of allowable calls to subroutine linegs (i/p)
*		for this particular variable F in this nonlinear iteration
*	ISOPT	option for linear equation solver: 1 implies line
*		Gauss-Seidel, 3 implies GMRES (i/p)
*	LSWPWE,LSWPSN,LSWPBT
*		Logicals to determine whether sweeps in linegs should
*		be done in the x-, y-, and z-directions respectively
*		(i/p)
*	LPARL	= T means we solve the equations in Jacobi mode (i/p)
*	LTEST	= T means that the exact solution option is being used
*		(i/p)
*	LLEFT	Logical, = T implies that left preconditioning is used
*		in the GMRES linear solver. = F implies that right 
*		preconditioning is used in the GMRES linear solver (i/p)
*	CLABEL	character array to store name of the variable being
*		solved for (i/p)
*
*-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision FF(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FCOR(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision RESF(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AE(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AS(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AT(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AB(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AP(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision SU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision GAMF,URFF,TSOLV,THETA,CONVL,R_NORM
      integer NI,NJ,NK
      integer ILCNT,IEQN,ISPF,NSWPF,ISOPT
      LOGICAL LPARL,LSWPWE,LSWPSN,LSWPBT,LTEST,LLEFT
      double precision tmpXYZArr1(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision tmpXYZArr2(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision tmpXArr1(indexLow(1):indexHigh(1))
      double precision tmpXArr2(indexLow(1):indexHigh(1))
      integer NEQ
      CHARACTER*4 CLABEL(NEQ+2)

c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision urfinv

c*********************************************************************
c     Start :
c*********************************************************************
      ISPF = 0

*-----SET CORRECTION ARRAY TO ZERO FOR INITIAL GUESS--------------------

      CALL ARRASS(indexLow, indexHigh, FCOR,ZERO)

*-----RESET SOURCE TERM FOR CORRECTION FORM--------------------------

      CALL ARRCOP(indexLow, indexHigh, SU,RESF,ONE)

*-----UNDERRELAXATION------------------------------------------------

      URFINV = ONE/URFF
      CALL ARRCOP(indexLow, indexHigh, AP,AP,URFINV)

*-----SOLUTION OF DIFFERENCE EQUATIONS-------------------------------

      CALL LISOLV(indexLow, indexHigh,
     &     FCOR,
     &     AE,AW,AN,AS,AT,AB,AP,SU,
     $     GAMF,TSOLV,THETA,CONVL,R_NORM,
     $     NI,NJ,NK,
     &     ILCNT,ISPF,ISOPT,NSWPF,IEQN,
     $     LSWPWE,LSWPSN,LSWPBT,LLEFT,LTEST,
     &     tmpXYZArr1, tmpXYZArr2,
     &     tmpXArr1, tmpXArr2,
     $     NEQ, CLABEL)

*-----------------------------------------------------------------------
*     UPDATE F IF WE ARE SOLVING IN GAUSS-SEIDEL MODE BUT USING
*     CORRECTIONS
*-----------------------------------------------------------------------

      IF (.NOT.LPARL) THEN
         CALL ARRADD(indexLow, indexHigh,
     &        FF,FF,FCOR,ONE,ONE)
      ENDIF

      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:50  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
