      SUBROUTINE rqacc(IST,IEND,JST,JEND,KST,KEND,
     &pcell,wall,ffield,abskg,esrcg,volq,
     &qfluxe,qfluxw,qfluxn,qfluxs,qfluxt,qfluxb,
     &qince,qincw,qincn,qincs,qinct,qincb,
     &src,totsrc,maxitr,qaccu,iflag,iriter,qerr,xu,yv,zw) 
C***********************************************************************
C     SUBROUTINE RQACC - calculates incident flux accuracy and flux
C     divergence for Cartesian and cylindrical coordinates;
C     returns IFLAG
C***********************************************************************
      IMPLICIT NONE

      integer WALL,FFIELD,IFLAG,IRITER,MAXITR,I,J,K,IST,IEND,
     &JST,JEND,KST,KEND
      integer PCELL(IST:IEND,JST:JEND,KST:KEND)

      double precision delqe(IST:IEND)
      double precision delqw(IST:IEND)
      double precision delqn(IST:IEND)
      double precision delqs(IST:IEND)
      double precision delqt(IST:IEND)
      double precision delqb(IST:IEND)

      double precision qince(IST:IEND,JST:JEND,KST:KEND)
      double precision qincw(IST:IEND,JST:JEND,KST:KEND)
      double precision qincn(IST:IEND,JST:JEND,KST:KEND)
      double precision qincs(IST:IEND,JST:JEND,KST:KEND)
      double precision qinct(IST:IEND,JST:JEND,KST:KEND)
      double precision qincb(IST:IEND,JST:JEND,KST:KEND)
      double precision qfluxe(IST:IEND,JST:JEND,KST:KEND)
      double precision qfluxw(IST:IEND,JST:JEND,KST:KEND)
      double precision qfluxn(IST:IEND,JST:JEND,KST:KEND)
      double precision qfluxs(IST:IEND,JST:JEND,KST:KEND)
      double precision qfluxt(IST:IEND,JST:JEND,KST:KEND)
      double precision qfluxb(IST:IEND,JST:JEND,KST:KEND)

      double precision ABSKG(IST:IEND,JST:JEND,KST:KEND)
      double precision ESRCG(IST:IEND,JST:JEND,KST:KEND)
      double precision VOLQ(IST:IEND,JST:JEND,KST:KEND)
      double precision SRC(IST:IEND,JST:JEND,KST:KEND)
      double precision XU(IST:IEND)
      double precision YV(JST:JEND)
      double precision ZW(KST:KEND)
 
      double precision PI,PI1,SIG,SIG4,FLXERR,QRATe,QRATw,QRATn,QRATs
      double precision QRATt,QRATb,QERR,QACCU,TOTSRC,VOLV

      DATA PI/3.141593D0/, SIG/5.67D-08/

C     Initialize variables

      PI1 = 1.0 / PI
      SIG4 = 1.0 / (4.0 * SIG) 
      FLXERR = 0.0

      DO 100 I=IST,IEND
        delqe(I) = 0.0
        delqw(I) = 0.0
        delqn(I) = 0.0
        delqs(I) = 0.0
        delqt(I) = 0.0
        delqb(I) = 0.0
100   CONTINUE

C     Calculate incident and net fluxes to all walls, flux error

	  DO 130 K=KST,KEND
	    DO 120 J=JST,JEND

	      DO 105 I=IST,IEND

	        IF ((PCELL(I,J,K) .EQ. WALL) .OR. 
     1              (I.EQ.1) .OR. (I.EQ.IEND) .OR. (J.EQ.1) 
     2           .OR.(J.EQ.JEND) .OR. (K.EQ.1) .OR. 
     3           (K.EQ.KEND)) THEN

	          delqe(I) = qince(I,J,K) - qfluxe(I,J,K)
	          delqw(I) = qincw(I,J,K) - qfluxw(I,J,K)
	          delqn(I) = qincn(I,J,K) - qfluxn(I,J,K)
	          delqs(I) = qincs(I,J,K) - qfluxs(I,J,K)
	          delqt(I) = qinct(I,J,K) - qfluxt(I,J,K)
	          delqb(I) = qincb(I,J,K) - qfluxb(I,J,K)

	          qfluxe(I,J,K) = qince(I,J,K)
	          qfluxw(I,J,K) = qincw(I,J,K)
	          qfluxn(I,J,K) = qincn(I,J,K)
	          qfluxs(I,J,K) = qincs(I,J,K)
	          qfluxt(I,J,K) = qinct(I,J,K)
	          qfluxb(I,J,K) = qincb(I,J,K)

	        END IF

105	      CONTINUE

	      DO 110 I=IST,IEND

	        IF (((qince(I,J,K) .NE. 0.0).AND.(qincw(I,J,K) .NE. 0.0) 
     &         .AND.(qincn(I,J,K) .NE. 0.0).AND.(qincs(I,J,K) .NE. 0.0)
     &         .AND.(qinct(I,J,K) .NE. 0.0).AND.(qincb(I,J,K) .NE. 0.0))
     &         .AND.((PCELL(I,J,K) .EQ. WALL) .OR. 
     &           (I.EQ.1) .OR.(I.EQ.IEND) .OR. 
     &           (J.EQ.1) .OR.(J.EQ.JEND) .OR. 
     &           (K.EQ.1) .OR.(K.EQ.KEND))) THEN

	          QRATe = ABS(delqe(I) / qince(I,J,K))
	          QRATw = ABS(delqw(I) / qincw(I,J,K))
	          QRATn = ABS(delqn(I) / qincn(I,J,K))
	          QRATs = ABS(delqs(I) / qincs(I,J,K))
	          QRATt = ABS(delqt(I) / qinct(I,J,K))
	          QRATb = ABS(delqb(I) / qincb(I,J,K))

	          IF (QRATe .GT. FLXERR) THEN
	            FLXERR = QRATe
	            IF (FLXERR .GT. QACCU) IFLAG = 1
	          END IF
	          IF (QRATw .GT. FLXERR) THEN
	            FLXERR = QRATw
	            IF (FLXERR .GT. QACCU) IFLAG = 1
	          END IF
	          IF (QRATn .GT. FLXERR) THEN
	            FLXERR = QRATn
	            IF (FLXERR .GT. QACCU) IFLAG = 1
	          END IF
	          IF (QRATs .GT. FLXERR) THEN
	            FLXERR = QRATs
	            IF (FLXERR .GT. QACCU) IFLAG = 1
	          END IF
	          IF (QRATt .GT. FLXERR) THEN
	            FLXERR = QRATt
	            IF (FLXERR .GT. QACCU) IFLAG = 1
	          END IF
	          IF (QRATb .GT. FLXERR) THEN
	            FLXERR = QRATb
	            IF (FLXERR .GT. QACCU) IFLAG = 1
	          END IF

	        END IF

110	      CONTINUE

120	    CONTINUE
130	  CONTINUE

	QERR = FLXERR * 100.0

C	Calculation of total incident radiation flux to a cell
C	Note: this INORD summation scheme is valid only for S2 and S4
C	      approximations where the weights are equal. For S6 and 
C	      higher order approximations, something similar to the
C	      NFAC scheme used in RDOMC needs to be used to match the
C	      appropriate weights and CINTM's.

	IF (IFLAG.EQ.0 .OR. IRITER.EQ.MAXITR) THEN

C	Calculation of energy balance on cell volumes,
C	SRC() - W/m^3, VOLQ() - W/m^2, TOTSRC - Watts

	  TOTSRC = 0.0

	  DO 280 K=KST+1,KEND-1
	    DO 270 J=JST+1,JEND-1
	      DO 260 I=IST+1,IEND-1

                SRC(I,J,K) = 0.0
	        IF (PCELL(I,J,K) .EQ. FFIELD) THEN
	          SRC(I,J,K) = (ABSKG(I,J,K)) * VOLQ(I,J,K)
     1                - 4.0*PI * (ESRCG(I,J,K))
	            VOLV = (XU(I+1) - XU(I)) * (YV(J+1) - YV(J)) 
     1                   * (ZW(K+1) - ZW(K))
	          TOTSRC = TOTSRC + VOLV * SRC(I,J,K)
      	        END IF
c     output for validation
c                if((J.EQ.7).AND.(K.EQ.7))WRITE(6,*)I,ABS(SRC(I,J,K))

260	      CONTINUE
270	    CONTINUE
280	  CONTINUE

	END IF

1000	FORMAT (4X,'Max flux error was ',F7.3,'% at',I3,','
     1          ,I3,',',I3,' at face',I2)

	RETURN
	END

