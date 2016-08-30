   	SUBROUTINE RMIE (X,REFREL,NANG,S1,S2,QEXT,QSCA,QBACK)
C***********************************************************************
C  	SUBROUTINE RMIE CALCULATES AMPLITUDE SCATTERING MATRIX ELEMENTS
C  	AND EFFICIENCIES FOR EXTINCTION, TOTAL SCATTERING AND BACK 
C  	SCATTERING FOR A GIVEN SIZE PARAMETER AND RELATIVE REF. INDEX;
C	(ROUTINE IS FROM BOHREN AND HUFFMAN, "ABSORPTION AND SCATTERING 
C	OF LIGHT BY SMALL PARTICLES," 1983.)
C  	CODED SEPT 1992 BY BRAD ADAMS
C***********************************************************************
   	DIMENSION AMU(100),THETA(100),PI(100),TAU(100),PI0(100),PI1(100)
   	COMPLEX D(3000),Y,REFREL,XI,XI0,XI1,AN,BN,S1(200),S2(200)
    	DOUBLE PRECISION PSI0,PSI1,PSI,DN,DX
   	DX=X
   	Y=X*REFREL
C***********************************************************************
C  	SERIES TERMINATED AFTER NSTOP TERMS
C***********************************************************************
   	XSTOP=X+4.0*X**.3333+2.0
   	NSTOP=XSTOP
   	YMOD=CABS(Y)
   	NMX=AMAX1(XSTOP,YMOD)+15
    	DANG=1.570796327/FLOAT(NANG-1) 
   	DO 555 J=1,NANG
   	  THETA(J)=(FLOAT(J)-1.0)*DANG
 555	AMU(J)=COS(THETA(J))
C***********************************************************************
C  	LOGARITIMIC DERIVATIVE D(J) CALCULATED BY DOWNWARD 
C  	RECURRENCE BEGINNING WITH INITAL VALUE 0.0+I*0.0
C  	AT J= NMX
C***********************************************************************
    	D(NMX)=CMPLX(0.0,0.0)
    	NN=NMX-1
    	DO 120 N=1,NN
    	  RN=NMX-N+1
 120	D(NMX-N)=(RN/Y)-(1.0/(D(NMX-N+1)+RN/Y))
    	DO 666 J=1,NANG
    	  PI0(J)=0.0
 666	PI1(J)=1.0
    	NN=2*NANG-1
    	DO 777 J=1,NN
    	  S1(J)=CMPLX(0.0,0.0)
 777	S2(J)=CMPLX(0.0,0.0)
C***********************************************************************
C   	RICCATI-BESSEL FUNCTION WITH REAL ARGUMENT X
C   	CALCULATED BY UPWARD RECURRENCE
C***********************************************************************
    	PSI0=DCOS(DX)
    	PSI1=DSIN(DX)
    	CHI0=-SIN(X)
    	CHI1=COS(X)
    	APSI0=PSI0
    	APSI1=PSI1
        XI0=CMPLX(APSI0,-CHI0)
    	XI1=CMPLX(APSI1,-CHI1)
    	QSCA=0.0
    	N=1
 200	DN=N
    	RN=N
    	FN=(2.0*RN+1.0)/(RN*(RN+1.0))
    	PSI=(2.0*DN-1.0)*PSI1/DX-PSI0
    	APSI=PSI
    	CHI=(2.0*RN-1.0)*CHI1/X-CHI0
    	XI=CMPLX(APSI,-CHI)
    	AN=(D(N)/REFREL+RN/X)*APSI-APSI1
    	AN=AN/((D(N)/REFREL+RN/X)*XI-XI1)
    	BN=(REFREL*D(N)+RN/X)*APSI-APSI1
    	BN=BN/((REFREL*D(N)+RN/X)*XI-XI1)
    	QSCA=QSCA+(2.0*RN+1.0)*(CABS(AN)*CABS(AN)+CABS(BN)*CABS(BN))
    	DO 789 J=1,NANG
    	  JJ=2*NANG-J
    	  PI(J)=PI1(J)
     	  TAU(J)=RN*AMU(J)*PI(J)-(RN+1.0)*PI0(J)
    	  P=(-1.0)**(N-1)
    	  S1(J)=S1(J)+FN*(AN*PI(J)+BN*TAU(J))
    	  T=(-1.0)**N
    	  S2(J)=S2(J)+FN*(AN*TAU(J)+BN*PI(J))
    	  IF(J.EQ.JJ) GO TO 789
    	  S1(JJ)=S1(JJ)+FN*(AN*PI(J)*P+BN*TAU(J)*T)
    	  S2(JJ)=S2(JJ)+FN*(AN*TAU(J)*T+BN*PI(J)*P)
 789	CONTINUE
    	PSI0=PSI1
     	PSI1=PSI
    	APSI1=PSI1
    	CHI0=CHI1
    	CHI1=CHI
    	XI1=CMPLX(APSI1,-CHI1)
    	N=N+1
    	RN=N
    	DO 999 J=1,NANG
    	  PI1(J)=((2.0*RN-1.0)/(RN-1.0))*AMU(J)*PI(J)
    	  PI1(J)=PI1(J)-RN*PI0(J)/(RN-1.0)
 999	PI0(J)=PI(J)
    	IF (N-1-NSTOP) 200,300,300
 300	QSCA=(2.0/(X*X))*QSCA
    	QEXT=(4.0/(X*X))*REAL(S1(1))
    	QBACK=(4.0/(X*X))*CABS(S1(2*NANG-1))*CABS(S1(2*NANG-1))
C
     	RETURN
    	END 
