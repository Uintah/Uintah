c*********************************************************************
c
c
c*********************************************************************

      double precision FUNCTION root(X1,X2,XI,XACC,PAR1,PAR2)
c
c     Using a combination of Newton-Raphson and bisection,
c     find the root of a function bracketed between X1 and X2.
c     The initial guess for the root is XI.  
c     The root, returned as the function value ROOT,
c     will be refined until its accuracy is known with +/- XACC.
c
c     FNCD is a user supplied subroutine which returns both the
c     function value and the first derivative of the function.
c
c     PAR1,PAR2 ARE PARAMETERS WHICH MAY BE NEEDED IN 
c     THE COMPUTATION OF THE FUNCTION AND THE DERIVATIVE IN FNCD
c

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      double precision X1, X2, XI, XACC, PAR1, PAR2

c*********************************************************************
c     Locals :
c*********************************************************************
      integer MAXIT
      PARAMETER (MAXIT=100)

      double precision FL, FH, DF, XL, XH, SWAP, DX, DXOLD, F, TEMP
      integer J

c*********************************************************************
c     Start :
c*********************************************************************
      CALL FNCD(X1,FL,DF,PAR1,PAR2)
      CALL FNCD(X2,FH,DF,PAR1,PAR2)
      IF(FL*FH.GE.ZERO) THEN
       PRINT *,X1,FL,X2,FH
       PAUSE 'root in ROOT must be bracketed'
      END IF
      IF(FL.LT.ZERO) THEN
        XL = X1
        XH = X2
      ELSE
        XH = X1
        XL = X2
        SWAP = FL
        FL = FH
        FH = SWAP
      END IF
      ROOT = XI
      DXOLD = ABS(X2 - X1)
      DX = DXOLD
      CALL FNCD(ROOT,F,DF,PAR1,PAR2)
      DO 11 J = 1,MAXIT
        IF(((ROOT-XH)*DF-F)*((ROOT-XL)*DF-F).GE.0
     &       .OR. ABS(TWO*F).GT.ABS(DXOLD*DF) ) THEN
          DXOLD = DX
          DX = PT5*(XH-XL)
          ROOT = XL+DX
          IF (XL.EQ.ROOT) RETURN
        ELSE
          DXOLD = DX
          DX = F/DF
          TEMP = ROOT
          ROOT = ROOT - DX
          IF (TEMP.EQ.ROOT) RETURN
        END IF
        IF(ABS(DX).LT.XACC) RETURN
        CALL FNCD(ROOT,F,DF,PAR1,PAR2)
        IF(F.LT.ZERO) THEN
          XL = ROOT
          FL = F
        ELSE
          XH = ROOT
          FH = F
        END IF
   11 CONTINUE
      PAUSE 'ROOT exceeding maximum iterations'
      RETURN
      END


c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:50  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
