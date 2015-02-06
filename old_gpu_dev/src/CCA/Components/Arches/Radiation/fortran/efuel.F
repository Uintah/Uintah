C 
C 
C The MIT License
C 
C Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
C Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
C University of Utah.
C 
C License for the specific language governing rights and limitations under
C Permission is hereby granted, free of charge, to any person obtaining a 
C copy of this software and associated documentation files (the "Software"),
C to deal in the Software without restriction, including without limitation 
C the rights to use, copy, modify, merge, publish, distribute, sublicense, 
C and/or sell copies of the Software, and to permit persons to whom the 
C Software is furnished to do so, subject to the following conditions:
C 
C The above copyright notice and this permission notice shall be included 
C in all copies or substantial portions of the Software.
C 
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
C OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
C THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
C DEALINGS IN THE SOFTWARE.
C 
C
C 
C 
      SUBROUTINE EFUEL(OMEGA,TEMP,PkPa,PTOT,GC,SDWEAK,SD7,SD3)

      DOUBLE PRECISION SDWEAK,GDINV,GDDINV,OMEGA,PI,BE,C2,WM,GD,
     &PE,PTOT,OM1,OM2,OM3,OM4,COM1,COM2,COM3,COM4,TEMP,DINV,W1,
     &SDB,SDA,SDC

      DOUBLE PRECISION BCNT(4),ATOT(4),PkPa(6),GC(5),SD7(3,16),
     &SD3(3,32)

      INTEGER I,J

      IF(OMEGA.GT.5000..OR.OMEGA.LT.1125.)GO TO 100
      PI=3.14159
      BE=5.2412
      C2=1.4388
      WM=16.
      GD=5.94E-6*OMEGA*(TEMP/(273.*WM))**.5
      IF(OMEGA.GT.3400.)GO TO 50
      PE=PTOT+.3*PkPa(3)
      IF(OMEGA.GE.2625.)GO TO 200
      IF(OMEGA.GT.1450.)GO TO 100
      GO TO 300
C
C   CONTRIBUTION TO 2.4 MICRON BAND FROM (0000)-(0110), (0000)-(0011),
C   (0000)-(1001), AND (0000)-(0102) TRANS.  THE INTEGRATED BAND INTENSITIES
C   OF VINCENT-GEISSE (ANNALES DE PHYSIQUE SER.12, V. 10, 1955) HAVE
C   BEEN MULTIPLIED BY A FACTOR OF 4 AND THE LINE SPACING IS THAT
C   OF V4 FROM GRAY AND PENNER (JQSRT V. 5, 1965).
C
50    OM1=2914.2
      OM2=1526.0
      OM3=3020.3
      OM4=1306.2
      BCNT(1)=4123.0
      BCNT(2)=4216.3
      BCNT(3)=4313.2
      BCNT(4)=4546.0
      COM1=OM2+2.*OM4
      COM2=OM1+OM4
      COM3=OM3+OM4
      COM4=OM2+OM3
      ATOT(1)=.64*273./TEMP**(1.-EXP(-C2*COM1/TEMP))/
     2(1.-EXP(-C2*OM2/TEMP))/(1.-EXP(-C2*OM4/TEMP))**2
      ATOT(2)=17.6*273./TEMP*(1.-EXP(-C2*COM2/TEMP))/
     2(1.-EXP(-C2*OM1/TEMP))/(1.-EXP(-C2*OM4/TEMP))
      ATOT(3)=14.8*273./TEMP*(1.-EXP(-C2*COM3/TEMP))/
     2(1.-EXP(-C2*OM3/TEMP))/(1.-EXP(-C2*OM4/TEMP))
      ATOT(4)=5.04*273./TEMP*(1.-EXP(-C2*COM4/TEMP))/
     2(1.-EXP(-C2*OM2/TEMP))/(1.-EXP(-C2*OM3/TEMP))
      DINV=1./5.74
      GDINV=GC(3)*DINV
      GDDINV=GD*DINV
      SDWEAK=0.0
      DO 51 I=1,4
      SDWEAK=SDWEAK+2.*(OMEGA-BCNT(I))**2*(C2*BE/TEMP)**1.5*ATOT(I)
     2/PI**0.5*DINV**3*EXP(-C2*BE*DINV**2/TEMP*(OMEGA-BCNT(I))**2)
51    CONTINUE
      SDWEAK=SDWEAK*(TEMP/273.)
      RETURN
C
CONTRIBUTION TO 3.3 MICRON BAND FROM (0000)-(0010) TRANS.
C   REFER TO BROSMER AND TIEN, JQSRT V. 33, P. 521
200   CONTINUE
      GDINV=.00734*PE*(273./TEMP)**.5*EXP(1.02*(TEMP-273.)/273.)
      GDDINV=GD/9.4
      J=(OMEGA-2600.)/25.
      W1=2600.+25.*FLOAT(J)
      SDB=SD3(2,J)+(OMEGA-W1)/25.*(SD3(2,J+1)-SD3(2,J))
      IF(TEMP.GT.600.)GO TO 260
      SDA=SD3(1,J)+(OMEGA-W1)/25.*(SD3(1,J+1)-SD3(1,J))
      SDWEAK=SDA+(TEMP-290.)/310.*(SDB-SDA)
      IF(SDWEAK.LT.0.)SDWEAK=0.
      RETURN
260   SDC=SD3(3,J)+(OMEGA-W1)/25.*(SD3(3,J+1)-SD3(3,J))
      SDWEAK=SDB+(TEMP-600.)/250.*(SDC-SDB)
      IF(SDWEAK.LT.0.)SDWEAK=0.
      RETURN
C
CONTRIBUTION TO 7.7 MICRON BAND FROM (0000)-(0001) TRANS.
C    REFER TO BROSMER AND TIEN, JQSRT V. 33, P. 521.
300   CONTINUE
      GDINV=.0243*PE*(TEMP/273.)**.8
      GDDINV=GD/5.1
      J=(OMEGA-1100.)/25.
      W1=1100.+25.*FLOAT(J)
      SDB=SD7(2,J)+(OMEGA-W1)/25.*(SD7(2,J+1)-SD7(2,J))
      IF(TEMP.GT.600.)GO TO 360
      SDA=SD7(1,J)+(OMEGA-W1)/25.*(SD7(1,J+1)-SD7(1,J))
      SDWEAK=SDA+(TEMP-290.)/310.*(SDB-SDA)
      IF(SDWEAK.LT.0.)SDWEAK=0.
      RETURN
360   SDC=SD7(3,J)+(OMEGA-W1)/25.*(SD7(3,J+1)-SD7(3,J))
      SDWEAK=SDB+(TEMP-600.)/250.*(SDC-SDB)
      IF(SDWEAK.LT.0.)SDWEAK=0.
      RETURN
100   SDWEAK=0.0
      GDINV=1.
      GDDINV=1.
      RETURN
      END
