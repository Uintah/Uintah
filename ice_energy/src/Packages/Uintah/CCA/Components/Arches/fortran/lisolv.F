c*********************************************************************
c
c
c*********************************************************************

#if 0
#include <Packages/Uintah/CCA/Components/Arches/fortran/lisolv_fort.h>
#else
      SUBROUTINE lisolv(indexLow, indexHigh,
     &     PHI,
     &     AE,AW,AN,AS,AT,AB,AP,SU,
     $     GAMF,TSOLV,THETA,CONVL,R_NORM,
     &     NI,NJ,NK,
     &     ILCNT,ISPF,ISOPT,NSWPS,IEQN,
     $     LSWPWE,LSWPSN,LSWPBT,LLEFT,LTEST,
     &     RESF, OLD,
     &     tmpXArr1, tmpXArr2,
     $     NEQ, CLABEL)
*-----------------------------------------------------------------------
*     WRITTEN BY: SESHADRI KUMAR, DECEMBER 30, 1996
*     Last modified: Seshadri Kumar, August 13, 1998
*
*     This routine solves the system of linear equations, each expressed
*     as:
*     AP_p*PHI_p - summation(Ad_p*PHId) = SU_p
*     where the summation is over d=E,W,N,S,T,B, and is written for
*     all interior points p.  This is thus a seven-diagonal system
*     when the variable PHI is written as a one-dimensional vector.
*
*     Two options are provided.  One (LINEGS, with ISOPT=1) is a 
*     stationary iterative method which is iterated upon till 
*     convergence to the prescribed tolerance is achieved.  The other 
*     (GMRES, with ISOPT=3) is a nonstationary method which uses
*     the stationary method as a preconditioner.
*
*     Variable definitions:
*
*     Passed variables:
*
*	PHI	variable for which equation is being solved (i/p) 
*               (o/p)(=x in Ax=b).  PHI is the initial guess for PHI
*               on input, and is overwritten by its new value on output.
*	AE,AW,AN,AS,AT,AB,AP
*		coefficients of the seven-diagonal finite difference
*		equation (i/p) (=A in Ax=b)
*	SU	Source term for the PHI-equation (i/p) (=b in the previous
*               discussion).
*	GAMF	linear equation convergence tolerance, specified
*		in setup.f (i/p)
*	TSOLV	Total CPU time taken for equation solution (i/p and o/p).
*	THETA 	Stone implicit acceleration factor (i/p).
*	CONVL	round-off error in linear equation solver (o/p).
*	RATIO	Ratio of equation residual after linear equation solution
*		to residual before linear equation solution (o/p)
*	R_NORM	normalization factor for residual to determine
*		its closeness to zero in relative terms (i/p)
*		This is used if a user-specified normalization factor
*		is used for this evaluation instead of using the residual
*		at the start for the evaluation
*	IST,JST,KST,IEND,JEND,KEND
*		Locations of the starting and ending indices for the
*		interior points of the PHI-equation (i/p)
*	NI	Number of total points (including ghost points) in
*		the I-direction (X-direction)
*	NJ	Number of total points (including ghost points) in
*		the J-direction (Y- or R-direction)
*	NK	Number of total points (including ghost points) in
*		the K-direction (Z- or THETA-direction)
*	ILCNT	Total number of calls to subroutine linegs in the
*		entire calculation (i/p and o/p)
*	ISPF	Number of calls made to linegs in this call to pdef
*		(o/p)
*	ISOPT	option for linear equation solver: 1 implies line
*		Gauss-Seidel, 3 implies GMRES (i/p)
*	NSWPS	Total number of allowable calls to subroutine linegs
*		for this particular variable in this nonlinear 
*               iteration (i/p)
*	LSWPWE,LSWPSN,LSWPBT
*		Logicals to determine whether sweeps in linegs should
*		be done in the x-, y-, and z-directions, respectively
*		(i/p)
*	LLEFT	Logical, = T implies that left preconditioning is used
*		in the GMRES linear solver. = F implies that right 
*		preconditioning is used in the GMRES linear solver (i/p)
*       LTEST   Logical, = T implies that the exact solution option
*               is being used (i/p)
*	CLABEL	character array to store name of the variable being
*		solved for (i/p)
*
*     Local (temporary) variables:
*
*	RESF	Array to store the residual of the linear equation
*     	OLD 	Temporary array used for storing current solution.
*		If linear solver does not converge, we use old values.
*
*-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision PHI(indexLow(1):indexHigh(1), 
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
      double precision GAMF,TSOLV,THETA,CONVL,R_NORM
      integer NI,NJ,NK
      integer ILCNT,ISPF,ISOPT,NSWPS,IEQN
      LOGICAL*1 LSWPWE,LSWPSN,LSWPBT,LLEFT,LTEST
      double precision RESF(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision OLD(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision tmpXArr1(indexLow(1):indexHigh(1))
      double precision tmpXArr2(indexLow(1):indexHigh(1))
      integer NEQ
      CHARACTER*4 CLABEL(NEQ+2)

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer NSOL
      double precision RTOT0, RTOT, RATIO, RES_NORM

*-----------------------------------------------------------------------
*     EXECUTABLE STATEMENTS
*-----------------------------------------------------------------------

*-----STORE OLD VALUES-----------------------------------------------

      CALL ARRCOP(indexLow, indexHigh,
     & PASS3A(OLD)
     & PASS3B(OLD),
     & PASS3A(PHI)
     & PASS3B(PHI),ONE)

*-----------------------------------------------------------------------

      IF (ISOPT.EQ.1) THEN

         CALL RESCAL(indexLow, indexHigh, 
     & PASS3A(PHI)
     & PASS3B(PHI),
     & PASS3A(RESF)
     & PASS3B(RESF),
     & PASS3A(AE)
     & PASS3B(AE),
     & PASS3A(AW)
     & PASS3B(AW),
     & PASS3A(AN)
     & PASS3B(AN),
     & PASS3A(AS)
     & PASS3B(AS),
     & PASS3A(AT)
     & PASS3B(AT),
     & PASS3A(AB)
     & PASS3B(AB),
     & PASS3A(AP)
     & PASS3B(AP),
     & PASS3A(SU)
     & PASS3B(SU),
     &        RTOT0)
         DO 600 NSOL = 1,NSWPS
            ISPF = ISPF + 1
 WARNING: This does not match the linegs prototype, so I did nto finish
 the conversion - Steve
            CALL LINEGS(indexLow, indexHigh,
     & PASS3A(PHI)
     & PASS3B(PHI),
     & PASS3A(AE)
     & PASS3B(AE),
     & PASS3A(AW)
     & PASS3B(AW),
     & PASS3A(AN)
     & PASS3B(AN),
     & PASS3A(AS)
     & PASS3B(AS),
     & PASS3A(AT)
     & PASS3B(AT),
     & PASS3A(AB)
     & PASS3B(AB),
     & PASS3A(AP)
     & PASS3B(AP),
     & PASS3A(SU)
     & PASS3B(SU),
     $           TSOLV,THETA,CONVL,
     &           ILCNT,
     $           LSWPWE,LSWPSN,LSWPBT,
     &           tmpXArr1, tmpXArr2)
         CALL RESCAL(indexLow, indexHigh, 
     & PASS3A(PHI)
     & PASS3B(PHI),
     & PASS3A(RESF)
     & PASS3B(RESF),
     & PASS3A(AE)
     & PASS3B(AE),
     & PASS3A(AW)
     & PASS3B(AW),
     & PASS3A(AN)
     & PASS3B(AN),
     & PASS3A(AS)
     & PASS3B(AS),
     & PASS3A(AT)
     & PASS3B(AT),
     & PASS3A(AB)
     & PASS3B(AB),
     & PASS3A(AP)
     & PASS3B(AP),
     & PASS3A(SU)
     & PASS3B(SU),
     &        RTOT)
            RATIO = RTOT/RTOT0
            RES_NORM = RTOT/RTOT0
C            RES_NORM = RTOT/R_NORM
            IF (RES_NORM.LE.GAMF) GO TO 800
 600     CONTINUE

      ELSE IF (ISOPT.EQ.3) THEN

c**** WARNING : BB 4/5 .. COMMENTING OUT GMRES FOR NOW ********
c
c        GAMF2 = GAMF
c
c        CALL GMRES(PHI,AE,AW,AN,AS,AT,AB,AP,SU,
c    $        TSOLV,THETA,CONVL,GAMF2,R_NORM,RATIO,
c    $        NI,NJ,NK,ILCNT,ITER,NSWPS,
c    $        IST,JST,KST,IEND,JEND,KEND,
c    $        LSWPWE,LSWPSN,LSWPBT,LLEFT)
c        ISPF = ISPF + ITER
c
c**** END COMMENT
         
      ENDIF

*-----------------------------------------------------------------------

      IF ((RATIO.GT.ONE).AND..NOT.LTEST) THEN
         PRINT *,CLABEL(IEQN),'is diverging by: ',RATIO
     $        ,' USING OLD VALUES'
         CALL ARRCOP(indexLow, indexHigh,
     & PASS3A(PHI)
     & PASS3B(PHI),
     & PASS3A(OLD)
     & PASS3B(OLD),
     & ONE)
      ENDIF

*-----------------------------------------------------------------------

 800  RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:48  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
