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
      SUBROUTINE m_ECO(OMEGA,TEMP,GC,SDWEAK,GDINV,GDDINV)

      DOUBLE PRECISION AA,BB,CC,DD,EE,FF,GG,SMINUS,SPLUS,
     &SDWEAK,SDSTRG,OMEGA,B,ALPHA,A,OME,WX,WY,OMPRIM,T0,
     &C2,V,GAM,OMV,DELTA,D,OMVBAR,F1,F2,TEST,TEMP,WM,GD,
     &DINV,GDINV,GDDINV

      DOUBLE PRECISION GC(5)

      INTEGER J

      IF(OMEGA.LT.1600.OR.OMEGA.GT.2400.) GO TO 300
      B=1.93139
      ALPHA=260.
      A=.017485
      OME=2170.21
      WX=13.461
      WY=.0308
      OMPRIM=OME-2.*WX+3.25*WY
      T0=300.
      C2=1.4388

      WM=28.
      GD=5.94E-6*OMEGA*(TEMP/(273.*WM))**.5
      SDWEAK=1.D-99
      SDSTRG=1.D-99
      AA=ALPHA*B*C2/(A*(1.-EXP(-OMPRIM*C2/T0))**2)
      BB=(1.-EXP(-OMEGA*C2/TEMP))*(1.-EXP(-OMPRIM*C2/TEMP))**2
      CC=AA*BB*OMEGA/TEMP*T0/TEMP

      DO 101 J=1,20

      V=FLOAT(J-1)
      DD=(V+1.)*EXP(-V*OME*C2/TEMP)
      GAM=B-A*(V+1.)
      OMV=OME-2.*(V+1.)*WX+(3.*(V+1.)*(V+1.)+.25)*WY
      DELTA=A*(OMEGA-OMV)
      IF(GAM*GAM.LE.DELTA) GO TO 102
      D=2.*(GAM*GAM-DELTA)**.5
      OMVBAR=OMV*(1.-EXP(-OMV*C2/TEMP))
      F1=GAM-D/2.
      F2=GAM+D/2.
      EE=C2*GAM/(A*A*TEMP)
      FF=DEXP(EE*DELTA*(1.+.5*A/GAM))
      SMINUS=CC*DD/OMVBAR*ABS(F1)*FF*DEXP(-EE*2.*GAM*F1)
      SPLUS=CC*DD/OMVBAR*ABS(F2)*FF*DEXP(-EE*2.*GAM*F2)
      GG=SDWEAK
      SDWEAK=(SMINUS+SPLUS)/D+SDWEAK
      TEST=(SDWEAK-GG)/SDWEAK
      IF(TEST.LT..0001) GO TO 102
      SDSTRG=(SMINUS**.5+SPLUS**.5)/D+SDSTRG
101   CONTINUE
102   CONTINUE
      DINV=SDSTRG*SDSTRG/SDWEAK
      GDINV=GC(4)*DINV
      GDDINV=GD*DINV

C***EXPRESS S/D AT STP, AS IS K IN NASA SP-3080
      SDWEAK=SDWEAK*TEMP/273.
      RETURN
300   SDWEAK=0.

      RETURN
      END
