      SUBROUTINE EH2O(OMEGA,TEMP,GC,SD,PkPaL,SDWEAK)

      DOUBLE PRECISION OMEGA,SDWEAK,TEMP,TTEMP,
     &W1,WW,TT,T1,TW,WM,GD,D,B,DINV,GDINV,GDDINV,small,
     &XSTAR,XC,XD,YC,YD,Y

      DOUBLE PRECISION SD(2256),GC(5),PkPaL(6)

      DATA small/1.0d-10/

      INTEGER I,J

      IF (OMEGA.GE.9300..OR.OMEGA.LT.50.)GO TO 200

C     Calculate fine structure parameters for broadening

c      WM=18.0d0
c      GD=5.94E-6*OMEGA*(TEMP/(273.*WM))**.5
c      D=-2.294+.3004E-02*TEMP-.366E-06*TEMP**2
c      B=SIN(.0036*OMEGA-8.043)
c      DINV=EXP(.7941*B+D)
c      GDINV=GC(2)*DINV
c      GDDINV=GD*DINV

C     Read value from Ludwig's table and interpolate

      J=(OMEGA-25.)/25.
      TTEMP=TEMP
      IF(TEMP.GE.2500.) TEMP=2499.99
      IF(TEMP.LT.300.) TEMP=300.
      I=TEMP/500. +1
      IF(I.EQ.2.AND.TEMP.LT.600.) I=1
      W1=25.+25.*FLOAT(J)
      WW=(OMEGA-W1)/25.
      IF(I.GT.2) GO TO 75
      IF(I.EQ.1) TT=(TEMP-300.)/300.
      IF(I.EQ.2) TT=(TEMP-600.)/400.
      GO TO 100
75    T1=FLOAT(I-1)*500.
      TT=(TEMP-T1)/500.
100   TW=TT*WW
      SDWEAK=SD(6*(J-1)+I)*(1.-TT-WW+TW)
     & +SD(6*(J-1)+(I+1))*(TT-TW)
     & +SD(6*(J+1-1)+I)*(WW-TW)
     & +SD(6*(J+1-1)+I+1)*TW

      TEMP=TTEMP

C     Now computing the correction factor for curve of growth

c      IF (SDWEAK.GT.1.0E-06)THEN

c      XSTAR = SDWEAK*PkPaL(2)+small

c      XC=XSTAR/(1.0D0+XSTAR/4.0D0/GDINV)**.5
c      XD=1.7*GDDINV*(DLOG(1.0D0+(XSTAR/1.7D0/GDDINV)**2))**.5

c      YC=1.-(XC/XSTAR)**2
c      YD=1.-(XD/XSTAR)**2

c      Y=1./YC**2+1./YD**2-1.

c      SDWEAK=SDWEAK*((1.-(Y**(-.5)))**.5)

c      ENDIF

      RETURN

200   CONTINUE
      SDWEAK=0.
      GDINV=1.0D0
      GDDINV=1.0D0

      RETURN
      END










