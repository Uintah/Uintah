	SUBROUTINE rqash
C***********************************************************************
C 	RQASH CALCULATES ARRAYS OF MEAN ABORPTION AND SCATTERING 
C	EFFICIENCIES FOR FLY-ASH AS A FUNCTION OF PARTICLE SIZE AND 
C	TEMPERATURE. SPECTRAL EFFICIENCIES ARE CALCULATED BASED ON 
C	THE SIZE PARAMETER (X) AND RELATIVE REFRACTIVE INDEX (REFREL)
C 	FOR A GIVEN SPHERE REFRACTIVE INDEX, MEDIUM REFRACTIVE INDEX,
C 	RADIUS, AND FREE SPACE WAVELENGTH USING MIE THEORY.
C 	MEAN EFFICIENCIES ARE CALCULATED AS THE AVERAGE OF THE PLANCK
C	AND ROSSELAND MEAN ABSORPTION AND SCATTERING EFFICIENCIES.
C
C	SUBROUTINES USED: RMIE, PLANCK, ROSSEL	
C	CODED SEPT 1992 BY BRAD ADAMS.
C***********************************************************************
      PARAMETER (NTD = 19, NTT = 14)
      COMMON
     & /LOGR1/  LRADI,LDBRAD,LCCOEF,LRPART,LRSOOT,LRTURB,LRUSNK
     & /RQMEAN/ QACOAL(NTD,NTT),QSCOAL(NTD,NTT),QAASH(NTD,NTT),
     &          QSASH(NTD,NTT),RTDIA(NTD),RTTEMP(NTT)
     & /RUSNK/ UNCOAL,UKCOAL,UNASH,UKASH
      LOGICAL LRADI,LDBRAD,LCCOEF,LRPART,LRSOOT,LRTURB,LRUSNK
      COMPLEX REFREL,S1(200),S2(200)
      DIMENSION QA(50),QS(50),QE(50),QB(50)
      Data PI/3.1415927/
C***********************************************************************
C  	REFMED = (REAL) REFRACTIVE INDEX OF SURROUNDING MEDIUM
C  	REFRACTIVE INDEX OF THE SPHERE = REFRE-I*REFIM	 
C  	RADIUS (RAD) AND WAVELENGTH (WAVEL) MUST HAVE THE SAME UNITS!
C	X = SIZE PARAMETER
C	DEFAULT REFRACTIVE INDEX IS FOR GOODWIN & MITCHNER'S SPECTRAL
C	FLY-ASH DATA (INT J HEAT MASS TRANSFER, 32, PP.627-638, 1989),
C	IF LRUSNK IS TRUE USER INPUTS ARE USED
C***********************************************************************
	REFMED = 1.0
	NWVL = 27
	DIAM = 0.5
	DO 300 J=1,NTD
	  RAD = DIAM / 2.
	  TEMP = 250.0
	  DO 200 K=1,NTT
	    WAVEL = 0.25
C***********************************************************************
C	GOODWIN'S SPECTRAL FLY-ASH DATA IS TEMPERATURE DEPENDENT
C***********************************************************************
	    DO 100 I=1,NWVL
	      IF (LRUSNK) THEN
	        REFRE = UNASH
	        REFIM = UKASH
	      ELSE IF (I .EQ. 1) THEN
	        REFRE = 1.55
	        REFIM = 0.001
	      ELSE IF (I .EQ. 2) THEN
	        REFRE = 1.55
	        REFIM = 0.00003
	      ELSE IF (I .EQ. 3) THEN
	        REFRE = 1.55
	        REFIM = 0.0003
	      ELSE IF (I .EQ. 4) THEN
	        REFRE = 1.55
	        REFIM = 0.0004
	      ELSE IF (I .EQ. 5) THEN
	        REFRE = 1.55
	        REFIM = 0.0004
	      ELSE IF (I .EQ. 6) THEN
	        REFRE = 1.5
	        REFIM = 0.0003
	      ELSE IF (I .EQ. 7) THEN
	        REFRE = 1.5
	        REFIM = 0.0003
	      ELSE IF (I .EQ. 8) THEN
	        REFRE = 1.5
	        REFIM = 0.0004
	      ELSE IF (I .EQ. 9) THEN
	        REFRE = 1.5
	        REFIM = 0.0004
	      ELSE IF (I .EQ. 10) THEN
	        REFRE = 1.5
	        REFIM = 0.0008
	        IF (TEMP.GT.1290. .AND. TEMP.LE.1650.) REFIM = 0.001
	        IF (TEMP.GT.1650.) REFIM = 0.0012
	      ELSE IF (I .EQ. 11) THEN
	        REFRE = 1.45
	        REFIM = 0.002
	        IF (TEMP.GT.1290. .AND. TEMP.LE.1650.) REFIM = 0.0025
	        IF (TEMP.GT.1650.) REFIM = 0.003
	      ELSE IF (I .EQ. 12) THEN
	        REFRE = 1.45
	        REFIM = 0.006
	        IF (TEMP.GT.1290. .AND. TEMP.LE.1650.) REFIM = 0.0075
	        IF (TEMP.GT.1650.) REFIM = 0.009
	      ELSE IF (I .EQ. 13) THEN
	        REFRE = 1.40
	        REFIM = 0.009
	        IF (TEMP.GT.1290. .AND. TEMP.LE.1650.) REFIM = 0.012
	        IF (TEMP.GT.1650.) REFIM = 0.014
	      ELSE IF (I .EQ. 14) THEN
	        REFRE = 1.35
	        REFIM = 0.012
	        IF (TEMP.GT.1290. .AND. TEMP.LE.1650.) REFIM = 0.016
	        IF (TEMP.GT.1650.) REFIM = 0.019
	      ELSE IF (I .EQ. 15) THEN
	        REFRE = 1.3
	        REFIM = 0.015
	        IF (TEMP.GT.1290. .AND. TEMP.LE.1650.) REFIM = 0.02
	        IF (TEMP.GT.1650.) REFIM = 0.024
	      ELSE IF (I .EQ. 16) THEN
	        REFRE = 1.2
	        REFIM = 0.022
	        IF (TEMP.GT.1290. .AND. TEMP.LE.1650.) REFIM = 0.029
	        IF (TEMP.GT.1650.) REFIM = 0.035
	      ELSE IF (I .EQ. 17) THEN
	        REFRE = 1.0
	        REFIM = 0.08
	      ELSE IF (I .EQ. 18) THEN
	        REFRE = 0.9
	        REFIM = 0.5
	      ELSE IF (I .EQ. 19) THEN
	        REFRE = 1.1
	        REFIM = 0.9
	      ELSE IF (I .EQ. 20) THEN
	        REFRE = 1.40
	        REFIM = 0.9
	      ELSE IF (I .EQ. 21) THEN
	        REFRE = 1.7
	        REFIM = 0.9
	      ELSE IF (I .EQ. 22) THEN
	        REFRE = 2.0
	        REFIM = 0.9
	      ELSE IF (I .EQ. 23) THEN
	        REFRE = 2.3
	        REFIM = 0.4
	      ELSE IF (I .EQ. 24) THEN
	        REFRE = 2.1
	        REFIM = 0.1
	      ELSE IF (I .EQ. 25) THEN
	        REFRE = 1.8
	        REFIM = 0.06
	      ELSE IF (I .EQ. 26) THEN
	        REFRE = 1.7
	        REFIM = 0.15
	      ELSE IF (I .GE. 27) THEN
	        REFRE = 1.7
	        REFIM = 0.20
	      END IF
   	      REFREL=CMPLX(REFRE,REFIM)/REFMED
   	      X=2.0*PI*RAD*REFMED/WAVEL
C***********************************************************************
C  	NANG = NUMBER OF ANGLES BETWEEN 0 AND 90 DEGREES
C  	MATRIX ELEMENTS CALCULATED AT 2*NANG-1 ANGLES
C  	INCLUDING 0, 90, AND 180 DEGREES
C***********************************************************************
   	      NANG=11
   	      DANG=1.570796327/FLOAT(NANG-1)
	      X = MIN(X,1500.) 
   	      CALL RMIE(X,REFREL,NANG,S1,S2,QEXT,QSCA,QBACK)
	      QA(I) = QEXT - QSCA
	      QS(I) = QSCA
	      IF (I .EQ. 1) WAVEL = WAVEL + 0.250
	      IF (I .GT. 1) WAVEL = WAVEL + 0.500
100	    CONTINUE
C***********************************************************************
C	NOW CALCULATE PLANCK AND ROSSELAND MEAN EFFICIENCIES; 
C	QASH'S ARE AVERAGE OF PLANCK & ROSSELAND MEANS
C***********************************************************************
	    CALL PLANCK(TEMP,QA,QAP)
	    CALL PLANCK(TEMP,QS,QSP)
	    CALL ROSSEL(TEMP,QA,QAR)
	    CALL ROSSEL(TEMP,QS,QSR)
	    QAASH(J,K) = (QAP + QAR) / 2.0
	    QSASH(J,K) = (QSP + QSR) / 2.0
	    RTTEMP(K) = TEMP
	    IF (K.LE.9) THEN
	      TEMP = TEMP + 150.0
	    ELSE IF (K.GT.9 .AND. K.LE.11) THEN
	      TEMP = TEMP + 200.0
	    ELSE IF (K.GT.11) THEN
	      TEMP = TEMP + 500.0
	    END IF
200	  CONTINUE
c	write (6,*) 'ash diam =',diam
c	write (6,*) 'qaash:'
c	write (6,*) (qaash(j,n),n=1,ntbin)
c	write (6,*) 'qsash:'
c	write (6,*) (qsash(j,n),n=1,ntbin)
	  RTDIA(J) = DIAM
	  IF (J.LE.2) THEN
	    DIAM = 2.0 * DIAM
	  ELSE IF (J.GT.2 .AND. J.LE.6) THEN
	    DIAM = DIAM + 2.0
	  ELSE IF (J.GT.6 .AND. J.LE.10) THEN
	    DIAM = DIAM + 2.5
	  ELSE IF (J.GT.10 .AND. J.LE.12) THEN
	    DIAM = DIAM + 5.0
	  ELSE IF (J.GT.12 .AND. J.LE.14) THEN
	    DIAM = DIAM + 10.0
	  ELSE IF (J.GT.14 .AND. J.LE.16) THEN
	    DIAM = DIAM + 25.0
	  ELSE IF (J.GT.16) THEN
	    DIAM = DIAM + 100.0
	  END IF
300	CONTINUE
C
	RETURN
   	END
