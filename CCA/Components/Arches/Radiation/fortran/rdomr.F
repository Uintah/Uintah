#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomr_fort.h>
C***********************************************************************
C	CALCULATION OF RADIATIVE HEAT TRANSFER IN 3-DIMENSIONAL 
C	CARTESIAN GEOMETRY USING THE DISCRETE ORDINATES METHOD
C	Original version written (5/2/94) by Brad Adams.
C       Modified by Gautham 03/02.
C	Definition of primary variables:
C	  ABSKG - gas absorption coefficients
C	  AF - differencing factor for diamond differencing scheme
C	  CINTM - intensity at the cell center
C	  OETA,OMU,OXI - direction cosines for the discrete ordinates
C	  WT - fraction of unit sphere surface area represented by each 
C	       discrete direction, intensity constant over this area
C	  QERR - flux convergence error
C	  QW,QINC - radiant fluxes at cell boundaries (arrays store old 
C	            and new versions of the same values)
C	  VESRCG - volumetric gas emission source term
C	  VOLQ - incident flux at a volume center (W/m^2)
C	  XINTBC,YINTBC,ZINTBC - intensities at 'upstream' cell faces
C	  XINTFC,YINTFC,ZINTFC - intensities at 'downstream' cell faces
C***********************************************************************

        double precision PI,PI1,qerrst,SIG,AREAEW

        integer negint,I,J,K,L,II,JJ,KK,IST,IEND,JST,JEND,KST,KEND
	
        DATA SIG/5.67d-08/

      pi = dacos(-1.0D0)
      IEND=idxhi(1)+1
      JEND=idxhi(2)+1
      KEND=idxhi(3)+1
       IST=idxlo(1)-1
       JST=idxlo(2)-1
       KST=idxlo(3)-1

C	Set constants and initial flags; AF is differencing factor: 
C	  1.0 - fully implicit (linear), more accurate
C	  0.0 - explicit (upwind), always stable; always start with 1

	AF = 0.0
	PI1 = 1. / PI
	iriter = 0
        qerrst = 0.0

C	Start of global iteration for radiation calculations.

C	Flux initializations

100     iriter = iriter + 1
101	iflag = 0
	negint = 0

	CALL RDOMINIT(IST,IEND,JST,JEND,KST,KEND,qince,qincw,qincn,
     &qincs,qinct,qincb,volq)

C	Loop to sweep through all flux approximation (quadrature) 

	  DO 450 L=1,sn*(sn+2)

C	K-loop - picks K-plane, performs T-B (z) sweeps, 
C	         direction depends on current xi (N)
 
	  DO 400 K=idxlo(3),idxhi(3)

C	RZIBC sets intensity boundary conditions for current K-plane

	  CALL RZIBC(K,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,tg,
     &zintbc,zintfc,pcell,wall,symtry,ffield,pfield,sfield,
     &pbcfld,abskg,oxi,qfluxt,qfluxb)

C	J-loop - picks J-row, performs N-S (y) sweeps,
C	         direction depends on current eta (M)

	  DO 350 J=idxlo(2),idxhi(2)

C	RYIBC sets intensity boundary conditions for current J-row

	  CALL RYIBC(J,JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,tg,
     &yintbc,yintfc,pcell,wall,symtry,ffield,pfield,sfield,
     &pbcfld,abskg,oeta,qfluxn,qfluxs)  

C       Load vectorizable area,volume arrays and vesrcg

c          AREAEW = RA(JJ) * SNS(JJ) * STB(KK)
c In bcenthalpy.F, RA(JJ) has been removed, therefore it is removed 
c here also.

           AREAEW = sns(JJ)*stb(KK)

C	I-loop - picks I-cell, performs E-W (x) sweeps thru I
C	         in given J-row, direction depends on mu (L)

C	Sweep from I=NI side (from East to West)

	          DO 300 I=idxlo(1),idxhi(1)

      CALL RXIBC(I,II,JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,tg,
     &xintbc,xintfc,pcell,wall,symtry,ffield,pfield,sfield,
     &pbcfld,abskg,omu,qfluxe,qfluxw)

      CALL RDOMCINTM(II,JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,
     &sew,sns,stb,AREAEW,omu,oeta,oxi,sn,abskg,esrcg,pcell,
     &ffield,AF,xintbc,xintfc,yintbc,yintfc,zintbc,zintfc,cintm,
     &NEGINT)

C     Calculate flux incident on East (2) and West (1) cell faces

      CALL RQIEW(II,JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,xintfc,
     &qince,qincw,omu,wt)

300	          CONTINUE

C	Calculate incident flux at cell centers here 

	            DO 330 I=idxlo(1),idxhi(1)

	              volq(I,JJ,KK) = volq(I,JJ,KK) 
     1                              + cintm(I,JJ,KK) * wt(L)

330	            CONTINUE

      CALL RQINS(JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,YINTFC,
     &qincn,qincs,OETA,WT)

350	      CONTINUE

      CALL RQITB(KK,L,IST,IEND,JST,JEND,KST,KEND,sn,zintfc,
     &qinct,qincb,oxi,wt)

400	CONTINUE

450	CONTINUE


C	End of global iteration for radiation calcs; 
C	RQACC calculates incident flux accuracy for convergence check

      CALL RQACC(IST,IEND,JST,JEND,KST,KEND,
     &pcell,wall,ffield,abskg,esrcg,volq,
     &qfluxe,qfluxw,qfluxn,qfluxs,qfluxt,qfluxb,
     &qince,qincw,qincn,qincs,qinct,qincb,
     &src,totsrc,maxitr,qaccu,iflag,iriter,qerr,xu,yv,zw) 

	IF (iriter .LT. maxitr) THEN
          IF (iriter .GE. 2) THEN
            IF (qerr.GT.qerrst) THEN 
              qerrst = 100.0
            ELSE
              qerrst = qerr
            END IF
          ELSE
            qerrst = qerr
          END IF
	  IF (iflag .GT. 0) GO TO 100
	END IF

	RETURN
	END




