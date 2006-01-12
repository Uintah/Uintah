c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/omgcal_fort.h>
#else
      SUBROUTINE omgcal(indexLow, indexHigh, 
     &     PHI,SCF,RES,
     &     AE,AW,AN,AS,AT,AB,AP,SU,
     $     TRNPHI,RWL1,RWL2,RWLINF,
     $     IINF,JINF,KINF,IEQNIN,NTOTAL,
     $     IEQN,NITER,NIJK,IK,IJ,
     $     LCSCF)
C-----------------------------------------------------------------------
C     Written by: Seshadri Kumar, December 11, 1996
C
C     This routine calculates the following metrics:
C     1. The order-of-magnitude term, T(i,j,k), in the finite-difference 
C        equation for each variable and its L1 norm over the domain,
C     2. The L1,L2, and Linfinity norms over said domain of the metric 
C        ROT, defined as |RES(i,j,k)/T(i,j,k)|, pointwise, and
C     3. The "Scaling Factor", defined as 
C        SCF(i,j,k) = T(i,j,k)/AP(i,j,k).
C     The routine also returns the location of the infinity norm of rot
C     and the corresponding equation.
C
C     GENERAL EXPLANATION:
C
C	PHI	Variable for which equation is being solved, (i/p)
C	AE,AW,AN,AS,AT,AB,AP
C		Coefficients of the seven-diagonal finite difference
C		equation for variable PHI, (i/p)
C	SU	Source term for variable PHI, (i/p)
C	RES	Array containing the residual of the PHI-equation (i/p)
C	RWL1, RWL2, RWLINF
C		L1, L2 and L-infinity norms of the R/T residual (o/p)
C	TRNPHI	L1 norm of the vector of "order-of-magnitude" terms
C		in the PHI-equation (o/p)
C	IINF,JINF,KINF,IEQNIN
C		Co-ordinates of the location of the infinity norm of
C		the residual R/T, among all equations (IEQNIN indicates
C		which equation the inf. norm falls in) (i/p and o/p)
C	NTOTAL	Total number of interior points solved for in all 
C		equations (i/p and o/p)
C	IEQN	Equation number of the PHI-equation (i/p)
C	NITER	Count variable for number of calls made to subroutine
C		fixpt (i/p)
C	NIJK	Total number of points in domain (i/p)
C	IJ,IK	Arrays defined in setup_gmres.f (i/p)
C	IST,JST,KST,IEND,JEND,KEND
C		Locations of the starting and ending indices for the
C		interior points of the PHI-equation (i/p)
C	LCSCF	Logical, =T implies that the scaled form of the Newton
C		equations are being solved for in the nonlinear 
C               GMRES solution method (i/p)
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
      double precision SCF(*)
      double precision RES(indexLow(1):indexHigh(1), 
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
      double precision TRNPHI,RWL1,RWL2,RWLINF
      integer IINF,JINF,KINF,IEQNIN,NTOTAL,IEQN,NITER,NIJK
      integer IK(*),IJ(*)
      logical LCSCF

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      double precision D1,D2,D3,D4,D5,D6,D7,D8,SUM1,SUM2,TRUNC,ROT,ROT2
      integer i,j,k,IJK
      
c*********************************************************************
c     Start :
c*********************************************************************
      IST = indexLow(1) + 1
      JST = indexLow(2) + 1
      KST = indexLow(3) + 1
      IEND = indexHigh(1) - 1
      JEND = indexHigh(2) - 1
      KEND = indexHigh(3) - 1

C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------
      TRNPHI = ZERO
      DO 900 K = KST,KEND
         DO 800 J = JST,JEND
            DO 700 I = IST,IEND

C-----CALCULATE ORDER OF MAGNITUDE TERM, TRUNC--------------------------

               D1 = AE(I,J,K)*PHI(I+1,J,K)
               D2 = AW(I,J,K)*PHI(I-1,J,K)
               D3 = AN(I,J,K)*PHI(I,J+1,K)
               D4 = AS(I,J,K)*PHI(I,J-1,K)
               D5 = AT(I,J,K)*PHI(I,J,K+1)
               D6 = AB(I,J,K)*PHI(I,J,K-1)
               D7 = -AP(I,J,K)*PHI(I,J,K)
               D8 = SU(I,J,K)

               SUM1 = MAX(ZERO,D1) + MAX(ZERO,D2) + 
     &              MAX(ZERO,D3) + MAX(ZERO,D4) + 
     &              MAX(ZERO,D5) + MAX(ZERO,D6) +
     &              MAX(ZERO,D7) + MAX(ZERO,D8)
               SUM2 = MIN(ZERO,D1) + MIN(ZERO,D2) +
     &              MIN(ZERO,D3) + MIN(ZERO,D4) + 
     &              MIN(ZERO,D5) + MIN(ZERO,D6) +
     &              MIN(ZERO,D7) + MIN(ZERO,D8)
               TRUNC = MAX(SUM1,ABS(SUM2))

C-----CALCULATE L1 NORM OVER DOMAIN-------------------------------------

               TRNPHI = TRNPHI + ABS(TRUNC)
               
C-----CALCULATE RESIDUAL NORMALIZED BY TRUNC, ROT-----------------------

               IF (TRUNC.GT.ZERO) THEN
                  ROT = ABS(RES(I,J,K)/TRUNC)
               ELSE
c                  IF (NITER.GT.1)
c     $                 PRINT*,'ZERO TRUNC ENCOUNTERED IN EQN',IEQN
                  ROT = ABS(RES(I,J,K))
               ENDIF

               ROT2 = ROT*ROT
               RWL1 = RWL1 + ROT
               RWL2 = RWL2 + ROT2
               IF (ROT.GT.RWLINF) THEN
                  RWLINF = ROT
                  IINF = I
                  JINF = J
                  KINF = K
                  IEQNIN = IEQN
               ENDIF
               NTOTAL = NTOTAL + 1

C-----CALCULATE SCALING FACTOR------------------------------------------

               IJK = I + IJ(J) + IK(K) + (IEQN-1)*NIJK
               IF (LCSCF) SCF(IJK) = TRUNC/AP(I,J,K)

C-----------------------------------------------------------------------

 700        CONTINUE
 800     CONTINUE
 900  CONTINUE

C-----------------------------------------------------------------------
C-----------------------------------------------------------------------
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:48  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
