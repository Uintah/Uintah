c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE rescal(indexLow, indexHigh,
     &     PHI,RES,
     &     AE,AW,AN,AS,AT,AB,AP,SU,
     &     RESPHI)
C-----------------------------------------------------------------------
C     WRITTEN BY: SESHADRI KUMAR, OCTOBER 14, 1996
C     Modified, December 11, 1996, to isolate TRNC calculation from
C     RES calculation, and to use matvec call for streamlining.
C
C     rescal calculates the pointwise residual of the finite difference
C     equation for any variable, between the limits (IST,JST,KST) and
C     (IEND,JEND,KEND) for that variable, and stores the residual in 
C     the array RES.  It also calculates the L1 norm of vector RES,
C     RESPHI.
C
C     EXPLANATION OF SYMBOLS USED:
C
C	AB 	Bottom coefficient (i/p)
C	AE	East coefficient in the seven-point stencil
C		equation for variable V (i/p)
C	AN	North coefficient (i/p)
C	AP	Diagonal coefficient (i/p)
C	AS	South coefficient (i/p)
C	AT	Top coefficient (i/p)
C	AW	West coefficient array (i/p)
C	IST,JST,KST
C		Starting indices for the calculation (i/p)
C	IEND,JEND,KEND
C		Ending indices for calculation (i/p)
C	PHI	Variable for which equation is being solved
C	RES	Array containing the residual of the PHI-equation (o/p)
C	RESPHI	L1 norm of the residual of the PHI-equation (o/p)
C	SU	Source term for the PHI-momentum equation,
C		excluding the pressure gradient (i/p)
C-----------------------------------------------------------------------
      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      double precision PHI(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision RES(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision AP(indexLow(1):indexHigh(1), 
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
      double precision SU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision RESPHI

c*********************************************************************
c     Local Variables :
c*********************************************************************
      double precision RESAVE

C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------
      CALL MATVEC(indexLow, indexHigh,
     &     PHI,RES,
     &     AE,AW,AN,AS,AT,AB,AP)
      CALL ARRADD(indexLow, indexHigh,
     &     RES,SU,RES,ONE,-ONE)
      CALL ARRL1(indexLow, indexHigh,
     &     RES,RESPHI,RESAVE)
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:50  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
