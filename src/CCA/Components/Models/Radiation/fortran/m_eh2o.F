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
      SUBROUTINE m_EH2O(OMEGA,TEMP,GC,SD,SDWEAK,GDINV,GDDINV)

      DOUBLE PRECISION OMEGA,SDWEAK,TEMP,TTEMP,
     &W1,WW,TT,T1,TW,WM,GD,D,B,DINV,GDINV,GDDINV

      DOUBLE PRECISION SD(2256),GC(5)

      INTEGER I,J

      IF (OMEGA.GE.9300..OR.OMEGA.LT.50.)GO TO 200

C     Calculate fine structure parameters for broadening

      WM=18.0d0
      GD=5.94E-6*OMEGA*(TEMP/(273.*WM))**.5
      D=-2.294+.3004E-02*TEMP-.366E-06*TEMP**2
      B=SIN(.0036*OMEGA-8.043)
      DINV=EXP(.7941*B+D)
      GDINV=GC(2)*DINV
      GDDINV=GD*DINV

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

      RETURN

200   CONTINUE
      SDWEAK=0.
      GDINV=1.0D0
      GDDINV=1.0D0

      RETURN
      END










