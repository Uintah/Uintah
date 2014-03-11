       SUBROUTINE ROSSEL(TEMP,Q,QR)
C***********************************************************************
C             SUBROUTINE ROSSEL CALCULATES THE ROSSELAND MEAN EFFICIENCIES
C       USING A REDUCED FORM OF THE PLANCK BB INTENSITY DISTRIBUTION
C       (SEE MENGUC & VISKANTA, COMBUST SCI TECH, 44, 143-159, 1985);
C       INTEGRATION IS PERFORMED USING SIMPSON'S COMPOSITE ALGORITHM
C***********************************************************************
       DIMENSION Q(50)
       DATA PI/3.1415927/,C2/14400.0/
C***********************************************************************
C       USE SMALL AND LARGE ZETA LIMITS TO APPROXIMATE 0-INF. INTEGRAL,
C       0.01-50.0 SEEMS TO APPROXIMATE THIS WELL WITHOUT RUNNING INTO
C       LARGE NUMBER PROBLEMS WITH EXPONENTIALS - PROPERTY RANGES ARE
C       ASSUMED CONSTANT BEYOND CALCULATED LIMITS TO ALLOW THIS;
C       C2DT IS 2ND RADIATION CONSTANT DIVIDED BY TEMPERATURE - THIS IS
C       USED TO EXTRACT CURRENT WAVELENGTH FOR DIFFERENT TEMPERATURES
C       (E.G., T=1000: C2DT=14.40, T=1500: C2DT=9.60, FOR SI UNITS)
C***********************************************************************
       A = 0.01
       B = 50.0
       C2DT = C2 / TEMP
       H = 0.01
       M = NINT((B - A) / (2. * H))
       FA = (1. / Q(27)) * A**4 * EXP(A) / ((EXP(A) - 1.)**2)
       FB = (1. / Q(1)) * B**4 * EXP(B) / ((EXP(B) - 1.)**2)
       XI0 = FA + FB
       XI1 = 0.0
       XI2 = 0.0
       DO 100 I=1,2*M-1
         Z = A + FLOAT(I)*H
         WL = C2DT / Z
         IF (WL.GE.12.75) THEN
           IW = 27
         ELSE IF (WL.LT.12.75 .AND. WL.GE.12.25) THEN
           IW = 26
         ELSE IF (WL.LT.12.25 .AND. WL.GE.11.75) THEN
           IW = 25
         ELSE IF (WL.LT.11.75 .AND. WL.GE.11.25) THEN
           IW = 24
         ELSE IF (WL.LT.11.25 .AND. WL.GE.10.75) THEN
           IW = 23
         ELSE IF (WL.LT.10.75 .AND. WL.GE.10.25) THEN
           IW = 22
         ELSE IF (WL.LT.10.25 .AND. WL.GE.9.75) THEN
           IW = 21
         ELSE IF (WL.LT.9.75 .AND. WL.GE.9.25) THEN
           IW = 20
         ELSE IF (WL.LT.9.25 .AND. WL.GE.8.75) THEN
           IW = 19
         ELSE IF (WL.LT.8.75 .AND. WL.GE.8.25) THEN
           IW = 18
         ELSE IF (WL.LT.8.25 .AND. WL.GE.7.75) THEN
           IW = 17
         ELSE IF (WL.LT.7.75 .AND. WL.GE.7.25) THEN
           IW = 16
         ELSE IF (WL.LT.7.25 .AND. WL.GE.6.75) THEN
           IW = 15
         ELSE IF (WL.LT.6.75 .AND. WL.GE.6.25) THEN
           IW = 14
         ELSE IF (WL.LT.6.25 .AND. WL.GE.5.75) THEN
           IW = 13
         ELSE IF (WL.LT.5.75 .AND. WL.GE.5.25) THEN
           IW = 12
         ELSE IF (WL.LT.5.25 .AND. WL.GE.4.75) THEN
           IW = 11
         ELSE IF (WL.LT.4.75 .AND. WL.GE.4.25) THEN
           IW = 10
         ELSE IF (WL.LT.4.25 .AND. WL.GE.3.75) THEN
           IW = 9
         ELSE IF (WL.LT.3.75 .AND. WL.GE.3.25) THEN
           IW = 8
         ELSE IF (WL.LT.3.25 .AND. WL.GE.2.75) THEN
           IW = 7
         ELSE IF (WL.LT.2.75 .AND. WL.GE.2.25) THEN
           IW = 6
         ELSE IF (WL.LT.2.25 .AND. WL.GE.1.75) THEN
           IW = 5
         ELSE IF (WL.LT.1.75 .AND. WL.GE.1.25) THEN
           IW = 4
         ELSE IF (WL.LT.1.25 .AND. WL.GE.0.75) THEN
           IW = 3
         ELSE IF (WL.LT.0.75 .AND. WL.GT.0.25) THEN
           IW = 2
         ELSE IF (WL.LE.0.25) THEN
           IW = 1
         END IF
         FX = (1. / Q(IW)) * Z**4 * EXP(Z) / ((EXP(Z) - 1.)**2)
         IF (MOD(I,2).EQ.0) THEN
           XI2 = XI2 + FX
         ELSE
           XI1 = XI1 + FX
         END IF
100    CONTINUE
       QI = H * (XI0 + 2.*XI2 + 4.*XI1) / 3.0
       QR = 1.0 / ((15. / (4. * PI**4)) * QI)
C
       RETURN
       END

