	SUBROUTINE rqpart(DIA,TMP,OMEGAA,AFRAC0,QABS,QSCA)
C***********************************************************************
C	RQPART USES CURRENT PARTICLE DIAMETER, TEMPERATURE AND ASH MASS
C	FRACTION TO INTERPOLATE VALUES OF MEAN ABSORPTION AND SCATTERING 
C	EFFICIENCIES FROM RADIATIVE PROPERTY TABLES. TABLES HAVE BEEN 
C	PREVIOUSLY CALCULATED IN ROUTINES RQCOAL AND RQASH. THESE
C	EFFICIENCIES ARE USED ALONG WITH PARTICLE SIZE AND NUMBER 
C	DENSITY INFORMATION TO CALCULATE PARTICLE ABSORPTION AND 
C	AND SCATTERING COEFFICIENTS IN CALLING ROUTINES.
C***********************************************************************
	INCLUDE 'param3.h'
	PARAMETER (NTD = 19, NTT = 14)
	COMMON
     & /RNSCAT/ COSPHI(NRQC,2,NRQ,NRQC,2,NRQ),NSCAT
     & /RQMEAN/ QACOAL(NTD,NTT),QSCOAL(NTD,NTT),QAASH(NTD,NTT),
     &          QSASH(NTD,NTT),RTDIA(NTD),RTTEMP(NTT)
C----------------------------------------------------------------------
C	Calculation of particle absorption and scattering coefficients
C----------------------------------------------------------------------
        PTMP = MAX(TMP,RTTEMP(1))
	PTMP = MIN(PTMP,RTTEMP(NTT))
       IF (PTMP.EQ.RTTEMP(NTT)) THEN
         NT = NTT-1
         T1 = 1.0
         T2 = 0.0
       ELSE
         DO 100 K=1,NTT-1
           IF (PTMP.GE.RTTEMP(K) .AND. PTMP.LT.RTTEMP(K+1)) THEN
             NT = K
             T1 = (PTMP - RTTEMP(K)) / (RTTEMP(K+1) - RTTEMP(K))
             T2 = (RTTEMP(K+1) - PTMP) / (RTTEMP(K+1) - RTTEMP(K))
             GOTO 105
           END IF
100      CONTINUE
       END IF
105    PD = 1.0E6 * DIA
       PDIA = MAX(PD,RTDIA(1))
       PDIA = MIN(PDIA,RTDIA(NTD))
       IF (PDIA.EQ.RTDIA(NTD)) THEN
         ND = NTD-1
         D1 = 1.0
         D2 = 0.0
       ELSE
         DO 200 J=1,NTD-1
           IF (PDIA.GE.RTDIA(J) .AND. PDIA.LT.RTDIA(J+1)) THEN
             ND = J
             D1 = (PDIA - RTDIA(J)) / (RTDIA(J+1) - RTDIA(J))
             D2 = (RTDIA(J+1) - PDIA) / (RTDIA(J+1) - RTDIA(J))
             GOTO 205
           END IF
200      CONTINUE
       END IF
205	OMEGA = MAX(OMEGAA,AFRAC0)
	A1 = (OMEGA - AFRAC0) / (1.0 - AFRAC0)
	A2 = (1.0 - OMEGA) / (1.0 - AFRAC0)
	QC1 = D1 * QACOAL(ND+1,NT) + D2 * QACOAL(ND,NT)
	QC2 = D1 * QACOAL(ND+1,NT+1) + D2 * QACOAL(ND,NT+1)
	QCT = T1 * QC2 + T2 * QC1
	QA1 = D1 * QAASH(ND+1,NT) + D2 * QAASH(ND,NT)
	QA2 = D1 * QAASH(ND+1,NT+1) + D2 * QAASH(ND,NT+1)
	QAT = T1 * QA2 + T2 * QA1
	QABS = A1 * QAT + A2 * QCT
	QC1 = D1 * QSCOAL(ND+1,NT) + D2 * QSCOAL(ND,NT)
	QC2 = D1 * QSCOAL(ND+1,NT+1) + D2 * QSCOAL(ND,NT+1)
	QCT = T1 * QC2 + T2 * QC1
	QA1 = D1 * QSASH(ND+1,NT) + D2 * QSASH(ND,NT)
	QA2 = D1 * QSASH(ND+1,NT+1) + D2 * QSASH(ND,NT+1)
	QAT = T1 * QA2 + T2 * QA1
	QSCA = A1 * QAT + A2 * QCT
C	For large particle assumption we use backward scattering and
C	must remove diffraction contribution to scattering efficiency
C	(which Mie theory includes). We assume the scattering efficiency
C	due to diffraction is 1.0 and scattering due to reflection is
C	everything else.
	IF (NSCAT.EQ.3) QSCA = MAX(0.0,(QSCA - 1.0))

	RETURN
	END

