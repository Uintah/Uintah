C      REAL FUNCTION GAMMALN(X)
      DOUBLE PRECISION FUNCTION DGAMMALN(X)
C------------------------------------------------------------------------------
C
C  This routine calculates the natural log of the gamma function for a 
C     real argument X.  The reason for calculating the natural log of 
C     the gamma function instead of the gamma function, is that gamma(171)
C     = 7.257E306.  Anything bigger that 171 will cause overflow which 
C     limits the range of X.  In order to calculate a larger range of X,
C     the natural log of the gamma function is used.  Gammaln(171) = 706.573.
C
C*******************************************************************************
C*******************************************************************************
C
C  References: Mathcad 8 Professional by MathSoft, Inc.
C
C  Author: Wing K. Yee
C
C  Last modified:  Sept 15, 1999
C
C------------------------------------------------------------------------------
      INTEGER J, K
c      REAL
      DOUBLE PRECISION
     1    X, FL, FLTERM1, FLTERM2, LG
      DOUBLE PRECISION CF(0:4)
C------------------------------------------------------------------------------
C  Check to see if x > 0
C------------------------------------------------------------------------------
      IF (X .LT. 0.0D9) THEN
         GAMMA = -1.0D0
         RETURN
      ELSE
C------------------------------------------------------------------------------
C  Initializing constants
C------------------------------------------------------------------------------
      CF(0) = 12.0D0
      CF(1) = -360.0D0
      CF(2) = 1260.0D0
      CF(3) = -1680.0D0
      CF(4) = 1188.0D0
      FLTERM1 = 0.0D0
      FLTERM2 = 0.0D0
      FL = 0.0D0
C------------------------------------------------------------------------------
C  Calculating the summation of the fl(x) first term
C------------------------------------------------------------------------------
      DO 100 K = 0, 10
         FLTERM1 = FLTERM1-LOG(X+K)
 100  CONTINUE
C------------------------------------------------------------------------------
C  Calculating the summation of the fl(x) second term
C------------------------------------------------------------------------------
      DO 200 J = 0, 4
         FLTERM2 = FLTERM2+(1.0D0/CF(J))*(1.0D0/((X+11.0D0)**
     *(1.0D0+2.0D0*J)))
 200  CONTINUE
C------------------------------------------------------------------------------
C  Calculating fl(x)
C------------------------------------------------------------------------------
      FL = FLTERM1+FLTERM2
C------------------------------------------------------------------------------
C  Calculating gammaln(x)
C------------------------------------------------------------------------------
      LG = FL+(X+10.5D0)*LOG(X+11.0D0)+9.189385332046727D-1-11.0D0-X
C------------------------------------------------------------------------------
C  Final adjustments and return
C------------------------------------------------------------------------------
C      GAMMALN = LG
      DGAMMALN = LG
      RETURN
      END IF
      END
