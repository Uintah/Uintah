      SUBROUTINE SOOT(OMEGA,FV,ABSKP)

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

      ABSKP= (36.0d0*3.1416*RIN*RIK*FV/LAMBDA)
     &/((RIN*RIN-RIK*RIK+2.0D0)**2+(2.0D0*RIN*RIK)**2)

      RETURN
      END
