C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the \"Software\"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
C 
      SUBROUTINE m_SOOT(OMEGA,FV,ABSKP)

      IMPLICIT NONE

      double precision OMEGA,FV,ABSKP,LAMBDA,RIN,RIK

c     omega is in (1/cm) convert to lambda(m)

      LAMBDA=1/(100*OMEGA)

C     ABSORPTION COEF. IS BASED UPON MEASUREMENTS OF DALZELL AND
C     SAROFIM,J.H.T, vol.91, #1, p.100-104,1969,(spectral)

      IF (LAMBDA.LT.0.4358E-06) THEN
         RIN = 1.57
         RIK = 0.46
      ELSE IF (LAMBDA.LT.0.4500E-06) THEN 
         RIN = 1.56
         RIK = 0.5
      ELSE IF (LAMBDA.LT.0.5500E-06) THEN 
         RIN = 1.57
         RIK = 0.53
      ELSE IF (LAMBDA.LT.0.6500E-06) THEN 
         RIN = 1.56
         RIK = 0.52
      ELSE IF (LAMBDA.LT.0.8065E-06) THEN 
         RIN = 1.57
         RIK = 0.49
      ELSE IF (LAMBDA.LT.2.5E-06) THEN 
         RIN = 2.04
         RIK = 1.15
      ELSE IF (LAMBDA.LT.3.0E-06) THEN 
         RIN = 2.21
         RIK = 1.23
      ELSE IF (LAMBDA.LT.4.0E-06) THEN 
         RIN = 2.38
         RIK = 1.44
      ELSE IF (LAMBDA.LT.5.0E-06) THEN 
         RIN = 2.07
         RIK = 1.72
      ELSE IF (LAMBDA.LT.6.0E-06) THEN 
         RIN = 2.62
         RIK = 1.67
      ELSE IF (LAMBDA.LT.7.0E-06) THEN 
         RIN = 3.05
         RIK = 1.91
      ELSE IF (LAMBDA.LT.8.5E-06) THEN 
         RIN = 3.26
         RIK = 2.10
      ELSE 
         RIN = 3.48
         RIK = 2.46
      END IF

c      ABSKP= (36.0d0*3.1416*RIN*RIK*FV/LAMBDA)
c     &/((RIN*RIN-RIK*RIK+2.0D0)**2+(2.0D0*RIN*RIK)**2)

      ABSKP = 7.0d0*FV/LAMBDA

      RETURN
      END
