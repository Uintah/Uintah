c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE linegs(indexLow, indexHigh,
     &     PHI,
     &     AE,AW,AN,AS,AT,AB,AP,SU,
     $     TSOLV,THETA,CONVL,
     &     ILCNT,
     $     LSWPWE,LSWPSN,LSWPBT,
     &     EE, FF)
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
*	TSOLV	Total CPU time taken for equation solution (i/p and o/p).
*	THETA 	Stone implicit acceleration factor (i/p).
*	CONVL	round-off error in linear equation solver (o/p).
*	ISTART,JSTART,KSTART
*		Locations of the starting indices for the
*		interior points of the PHI-equation (i/p)
*               (These points are the ones being solved for).
*	NI	Number of total points (including ghost points) in
*		the I-direction (X-direction).
*	NJ	Number of total points (including ghost points) in
*		the J-direction (Y- or R-direction).
*	NK	Number of total points (including ghost points) in
*		the K-direction (Z- or THETA-direction).
*	ILCNT	Total number of calls to subroutine linegs in the
*		entire calculation (i/p and o/p).  This variable is
*               incremented each time a call to linegs is made.
*	LSWPWE,LSWPSN,LSWPBT
*		Logicals to determine whether sweeps in linegs should
*		be done in the x-, y-, and z-directions respectively
*		(i/p).
*     
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
      double precision EE(indexLow(1):indexHigh(1)) 
      double precision FF(indexLow(1):indexHigh(1)) 
      double precision TSOLV,THETA,CONVL
      integer ILCNT
      LOGICAL LSWPWE,LSWPSN,LSWPBT

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k, ii, jj, kk
      integer ISTART, JSTART, KSTART, NI,NJ,NK
      integer ISTM1,JSTM1,KSTM1, NIM1,NJM1,NKM1, NKUSED
      double precision CONVWE, CONVSN, CONVBT, RHS, APC, DENOM
      double precision T3,T4

C     Note that theta is an under-relaxation factor (0-1)
C-----------------------------------------------------------------------
      ILCNT = ILCNT + 1
      CALL CPUTIM(T3)
      ISTART = indexLow(1)+1
      JSTART = indexLow(2)+1
      KSTART = indexLow(3)+1
      NI = indexHigh(1)
      NJ = indexHigh(2)
      NK = indexHigh(3)
      NIM1 = NI-1
      NJM1 = NJ-1
      NKM1 = NK-1
      ISTM1 = ISTART-1
      JSTM1 = JSTART-1
      KSTM1 = KSTART-1      
C-----------------------------------------------------------------------
C     west to east X DIRECTION SWEEP  
C-----------------------------------------------------------------------
      CONVWE = ZERO
      IF (LSWPWE) THEN
C-----------------------------------------------------------------------
C     COMMENCE S-N AND B-T TRAVERSE
C-----------------------------------------------------------------------
        DO 200 K = KSTART,NKM1
          DO 190 J = JSTART,NJM1
            EE(ISTM1) = ZERO
            FF(ISTM1) = PHI(ISTM1,J,K)
C----------------------------------------------------------------------- 
C     COMMENCE W-E SWEEP 
C-----------------------------------------------------------------------
            DO 140 I = ISTART,NIM1
C----------------------------------------------------------------------- 
C     ASSEMBLE TDMA COEFFICIENTS
C-----------------------------------------------------------------------
              RHS = AN(I,J,K)*PHI(I,J+1,K)+AS(I,J,K)*PHI(I,J-1,K)
     &             +AT(I,J,K)*PHI(I,J,K+1)+AB(I,J,K)*PHI(I,J,K-1)
     &             +SU(I,J,K)
              APC = THETA*(AN(I,J,K)+AS(I,J,K)+AT(I,J,K))
              RHS = RHS - APC*PHI(I,J,K)
C-----------------------------------------------------------------------
C     CALCULATE COEFFICIENTS OF RECURRENCE FORMULA
C-----------------------------------------------------------------------
              DENOM = ONE/(AP(I,J,K)-APC-AW(I,J,K)*EE(I-1))
              EE(I)  = AE(I,J,K)*DENOM
              FF(I)  = (RHS+AW(I,J,K)*FF(I-1))*DENOM
  140       CONTINUE
C-----------------------------------------------------------------------
C     OBTAIN NEW PHI'S
C-----------------------------------------------------------------------
            DO 180 II = ISTART,NIM1
              I = NI+ISTM1-II
              PHI(I,J,K) = (EE(I)*PHI(I+1,J,K))+FF(I)
              CONVWE = CONVWE + MIN(ABS(EE(I)*PHI(I+1,J,K)),
     &                 ABS(FF(I)))/MAX(TINY,
     &                 ABS(EE(I)*PHI(I+1,J,K)),ABS(FF(I)))
  180       CONTINUE
  190     CONTINUE
  200   CONTINUE
      END IF
C-----------------------------------------------------------------------
C       south to north R OR Y DIRECTION SWEEP   
C-----------------------------------------------------------------------
      CONVSN = ZERO
      IF (LSWPSN) THEN
C-----------------------------------------------------------------------
C     COMMENCE W-E AND B-T TRAVERSE
C-----------------------------------------------------------------------
        DO 400 K = KSTART,NKM1
          DO 390 I = ISTART,NIM1
            EE(JSTM1) = ZERO
            FF(JSTM1) = PHI(I,JSTM1,K)
C-----------------------------------------------------------------------
C     COMMENCE S-N SWEEP
C-----------------------------------------------------------------------
            DO 340 J = JSTART,NJM1
C-----------------------------------------------------------------------
C     ASSEMBLE TDMA COEFFICIENTS
C-----------------------------------------------------------------------
              RHS = AE(I,J,K)*PHI(I+1,J,K)+AW(I,J,K)*PHI(I-1,J,K)
     &             +AT(I,J,K)*PHI(I,J,K+1)+AB(I,J,K)*PHI(I,J,K-1)
     &             +SU(I,J,K)
              APC = THETA*(AE(I,J,K)+AW(I,J,K)+AT(I,J,K))
              RHS = RHS - APC*PHI(I,J,K)
C-----------------------------------------------------------------------
C     CALCULATE COEFFICIENTS OF RECURRENCE FORMULA
C-----------------------------------------------------------------------
              DENOM = ONE/(AP(I,J,K)-APC-AS(I,J,K)*EE(J-1))
              EE(J)  = AN(I,J,K)*DENOM
              FF(J)  = (RHS+AS(I,J,K)*FF(J-1))*DENOM              
  340       CONTINUE
C-----------------------------------------------------------------------
C     OBTAIN NEW PHI'S
C-----------------------------------------------------------------------
            DO 380 JJ = JSTART,NJM1
              J = NJ+JSTM1-JJ
              PHI(I,J,K) = EE(J)*PHI(I,J+1,K)+FF(J)
              CONVSN = CONVSN + MIN(ABS(EE(J)*PHI(I,J+1,K)),
     &                 ABS(FF(J)))/MAX(ABS(EE(J)*PHI(I,J+1,K)),
     &                 ABS(FF(J)),TINY)
  380       CONTINUE
  390     CONTINUE
  400   CONTINUE
      END IF 
C-----------------------------------------------------------------------
C     bottom to top Z OR THETA DIRECTION SWEEP  
C-----------------------------------------------------------------------
      CONVBT = ZERO
      IF (LSWPBT) THEN
C-----------------------------------------------------------------------
C     COMMENCE W-E AND S-N TRAVERSE
C-----------------------------------------------------------------------
        DO 600 J = JSTART,NJM1
          DO 590 I = ISTART,NIM1
            EE(KSTM1) = ZERO
            FF(KSTM1) = PHI(I,J,KSTM1)
C-----------------------------------------------------------------------
C     COMMENCE B-T SWEEP
C-----------------------------------------------------------------------
            DO 540 K = KSTART,NKM1
C-----------------------------------------------------------------------
C     ASSEMBLE TDMA COEFFICIENTS
C-----------------------------------------------------------------------
              RHS = AE(I,J,K)*PHI(I+1,J,K)+AW(I,J,K)*PHI(I-1,J,K)
     &             +AN(I,J,K)*PHI(I,J+1,K)+AS(I,J,K)*PHI(I,J-1,K)
     &             +SU(I,J,K)
              APC = THETA*(AE(I,J,K)+AW(I,J,K)+AN(I,J,K))
              RHS = RHS - APC*PHI(I,J,K)
C-----------------------------------------------------------------------
C     CALCULATE COEFFICIENTS OF RECURRENCE FORMULA
C-----------------------------------------------------------------------
              DENOM = ONE/(AP(I,J,K)-APC-AB(I,J,K)*EE(K-1))
              EE(K)  = AT(I,J,K)*DENOM
              FF(K)  = (RHS+AB(I,J,K)*FF(K-1))*DENOM
  540       CONTINUE
C-----------------------------------------------------------------------
C     OBTAIN NEW PHI'S
C-----------------------------------------------------------------------
            DO 580 KK = KSTART,NKM1
              K = NK+KSTM1-KK
              PHI(I,J,K) = EE(K)*PHI(I,J,K+1)+FF(K)
              CONVBT = CONVBT + MIN(ABS(EE(K)*PHI(I,J,K+1)),
     &                 ABS(FF(K)))/MAX(ABS(EE(K)*PHI(I,J,K+1)),
     &                 ABS(FF(K)),TINY)
  580       CONTINUE
  590     CONTINUE
  600   CONTINUE
      END IF
C
      NKUSED = NK - KSTART
      IF (NKUSED .LE. 0) NKUSED = 1
      CONVL = MAX(CONVWE,CONVSN,CONVBT)/
     &        (NKUSED*(NJ-JSTART)*(NI-ISTART))
      CALL CPUTIM(T4) 
      TSOLV = TSOLV + T4 - T3
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
