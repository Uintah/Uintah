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
#include <CCA/Components/Models/Radiation/fortran/m_radcal_fort.h>
C***********************************************************************
C	SUBROUTINE RADCAL - calculates spectral gas and soot absorption  
C	coefficients for cartesian or cylindrical geometries.
C***********************************************************************
C      IMPLICIT NONE 
	
      double precision PI,SIG,CTWO,PI1,PTOT,SDWEAK,UK, 
     &x,y,z,xxx,yyy,zzz,TEMP,ABSKP,FV,FACTOR,
     &MUONE,MUTWO,SUMONE,SUMTWO,FRACNONE,FRACNTWO,FRACN,twall,
     &small,wavemin,wavemax,dom,omega,SDTOTAL,GDINV,
     &GDDINV,XSTAR,XC,XD,YC,YD,YCD,scatcoef,phfunc,abskgtemp

      Integer I,II,J,K,L,M,intervals

      double precision DELTAX(idxlo(1)-1:idxhi(1)+1),
     &FRACNSUM(idxlo(1):idxhi(1),idxlo(2):idxhi(2),idxlo(3):idxhi(3)),
     &PATCHN(idxlo(1):idxhi(1),idxlo(2):idxhi(2),idxlo(3):idxhi(3)),
     &PATCHD(idxlo(1):idxhi(1),idxlo(2):idxhi(2),idxlo(3):idxhi(3)),
     &GC(5),PkPa(6),PkPaL(6)

      DATA PI/3.141593D0/,SIG/5.67E-08/,CTWO/1.4388d0/,small/1.0d-06/

      PI1 = 1.0d0 / PI

c     Selection of appropriate band intervals

      fraction(bands)= 0.0d0	
      dom = 25

c      if (lambda.eq.1)then
c      wavemin = 50
c      wavemax = 10000
c      intervals = 399
c      omega = wavemin - dom
c      end if

      if (bands.eq.1)then
      wavemin = 50
      wavemax = 1000
      intervals = 39
      omega = wavemin - dom

      else if (bands.eq.2)then
      wavemin = 1025
      wavemax = 2200
      intervals = 48
      omega = wavemin - dom

      else if (bands.eq.3)then
      wavemin = 2225
      wavemax = 2400
      intervals = 8
      omega = wavemin - dom

      else if (bands.eq.4)then
      wavemin = 2425
      wavemax = 3450
      intervals = 42
      omega = wavemin - dom

      else if (bands.eq.5)then
      wavemin = 3475
      wavemax = 3800
      intervals = 14
      omega = wavemin - dom

      else if (bands.eq.6)then
      wavemin = 3825
      wavemax = 10000
      intervals = 248
      omega = wavemin - dom

      end if

      go to 50

50    CONTINUE

      DO 80 K=idxlo(3),idxhi(3)
        DO 70 J=idxlo(2),idxhi(2)
          DO 60 I=idxlo(1),idxhi(1)

          IF (PCELL(I,J,K).EQ.FFIELD) THEN
            ABSKG(I,J,K) = 0.0D0
            ESRCG(I,J,K) = 0.0D0

	    FRACNSUM(I,J,K) = 0.0D0
	    PATCHN(I,J,K) = 0.0D0
	    PATCHD(I,J,K) = 0.0D0
          ENDIF

60        CONTINUE
70      CONTINUE
80    CONTINUE

C     LOOP 160 COMPUTES THE CONTRIBUTION OF EACH SPECIES TO TAU
C     *********************************************************

C          THE SPECIES CONSIDERED ARE
C          L   SPECIES
C          1     CO2
C          2     H2O
C          3     CH4
C          4     CO
C          5     PARTICULATES(SOOT)

        DO 200 M=1,intervals

                omega = omega + dom

          DO 190 K=idxlo(3),idxhi(3)
            DO 180 J=idxlo(2),idxhi(2)
              DO 170 I=idxlo(1),idxhi(1)

              IF (PCELL(I,J,K).EQ.FFIELD) THEN

                 ABSKP = 0.0d0
       		 SDTOTAL = 0.0d0              

                DO 160 L=1,4

                 SDWEAK  = 0.0d0
                 GC(L)   = 0.0d0

c     Partial Pressures of each specie in (atm)

                 PkPa(1) = CO2(I,J,K)
                 PkPa(2) = H2O(I,J,K)
                 PkPa(3) = 0.0d0
                 PkPa(4) = 0.0d0
                 PkPa(5) = 0.0d0
c                 PkPa(6) = 0.67d0
                 PkPa(6) = (1.0d0-(CO2(I,J,K)+H2O(I,J,K)))

          PkPaL(L)=(273.0d0/TG(I,J,K))*PkPa(L)*100.0d0*OPL

                 GC(L)=0.0d0
                 PTOT=0.0d0
                DO 110 II=1,6
                 PTOT=PkPa(II)+PTOT
                 GC(L)=GC(L)+RGAMMA(4*(II-1)+L)*PkPa(II)*
     &           (273.0d0/TG(I,J,K))**0.5
110             CONTINUE
                 GC(L)=GC(L)+RGAMMA(4*(7-1)+L)*PkPa(L)*273.0d0/TG(I,J,K)

111              IF(PkPa(L).EQ.0.0d0) GO TO 118
                 TEMP = TG(I,J,K)

                 GO TO(112,113,114,115),L

112          CALL m_ECO2(OMEGA,TEMP,GC,SD15,SDWEAK,GDINV,GDDINV)     
                 GO TO 116
113          CALL m_EH2O(OMEGA,TEMP,GC,SD,SDWEAK,GDINV,GDDINV)
                 GO TO 116
c114         CALL m_EFUEL(OMEGA,TEMP,PkPa,PTOT,GC,SDWEAK,SD7,SD3)
114              CONTINUE
                 GO TO 116
c115          CALL m_ECO(OMEGA,TEMP,GC,SDWEAK,GDINV,GDDINV)
115              CONTINUE

116              CONTINUE

                 IF (SDWEAK.LE.small)SDWEAK = small

      IF (LPLANCKMEAN) GO TO 117

      XSTAR = SDWEAK*PkPaL(L)
                 IF (XSTAR.LE.small)XSTAR = small

      XC=XSTAR/(1.0D0+XSTAR/4.0D0/GDINV)**.5
      XD=1.7*GDDINV*(DLOG(1.0D0+(XSTAR/1.7D0/GDDINV)**2))**.5

      YC=1.-(XC/XSTAR)**2
      YD=1.-(XD/XSTAR)**2

      YCD=1./YC**2+1./YD**2-1.

      SDWEAK=XSTAR*((1.-(YCD**(-.5)))**.5)/OPL

117   CONTINUE

      IF (LPLANCKMEAN)THEN

c     The following converts the absorption coefficient to (1/m)
c     Calculate the factor for equivalent path length

      IF (L.EQ.1) FACTOR = (273.0d0*CO2(I,J,K))/TEMP
      IF (L.EQ.2) FACTOR = (273.0d0*H2O(I,J,K))/TEMP
c      IF (L.EQ.4) FACTOR = (273.0d0*0.03)/TEMP

c     Convert to (1/cm)
                 SDWEAK = SDWEAK*FACTOR

c     Convert to (1/m)

                 SDWEAK = SDWEAK*100

      END IF

                 SDTOTAL = SDTOTAL+SDWEAK

118		CONTINUE

160             CONTINUE

                 FV = SFV(I,J,K)
                 CALL m_SOOT(OMEGA,FV,ABSKP)
                 SDTOTAL=SDTOTAL+ABSKP

C     Calculate blackbody emissive intensity (emissive power / pi)
C     Based on function for fractional emission in each waveband 
C     provided by Chang and Rhee (1984).

       MUONE = CTWO*OMEGA/TG(I,J,K)
       MUTWO = CTWO*(OMEGA+DOM)/TG(I,J,K)
       
       SUMONE = 0.0
       SUMTWO = 0.0

       DO 162 II=1,10

          SUMONE = SUMONE + (EXP(-II*MUONE)/II)*((MUONE**3) + 
     &(3*(MUONE**2)/II)+(6*MUONE/(II**2))+(6/(II**3)))

          SUMTWO = SUMTWO + (EXP(-II*MUTWO)/II)*((MUTWO**3) + 
     &(3*(MUTWO**2)/II)+(6*MUTWO/(II**2))+(6/(II**3)))

 162   CONTINUE

       FRACNONE = (15 * (PI1**4))*SUMONE
       FRACNTWO = (15 * (PI1**4))*SUMTWO

       FRACN = FRACNTWO - FRACNONE
       FRACNSUM(I,J,K) = FRACNSUM(I,J,K) + ABS(FRACN)

       ESRCG(I,J,K) = ESRCG(I,J,K) + ABS(FRACN)*
     & SIG*PI1*SDTOTAL*TG(I,J,K)**4


      IF (LPATCHMEAN)THEN

       PATCHN(I,J,K) = PATCHN(I,J,K) + (ABS(FRACN)*
     & SIG*PI1*SDTOTAL*TG(I,J,K)**4)*EXP(-SDTOTAL*OPL)
       PATCHD(I,J,K) = PATCHD(I,J,K) + (ABS(FRACN)*
     & SIG*PI1*TG(I,J,K)**4)*EXP(-SDTOTAL*OPL)

      END IF

              ENDIF

170           CONTINUE
180         CONTINUE
190       CONTINUE

c       Calculate fraction of the blackbody wall emission here (this is only 
c       valid when all walls are at same temperature)

       TWALL = 293.0d0
       
       MUONE = CTWO*OMEGA/TWALL
       MUTWO = CTWO*(OMEGA+DOM)/TWALL
       
       SUMONE = 0.0
       SUMTWO = 0.0

       DO 163 II=1,10

          SUMONE = SUMONE + (EXP(-II*MUONE)/II)*((MUONE**3) + 
     &(3*(MUONE**2)/II)+(6*MUONE/(II**2))+(6/(II**3)))

          SUMTWO = SUMTWO + (EXP(-II*MUTWO)/II)*((MUTWO**3) + 
     &(3*(MUTWO**2)/II)+(6*MUTWO/(II**2))+(6/(II**3)))

 163	CONTINUE

       FRACNONE = (15 * (PI1**4))*SUMONE
       FRACNTWO = (15 * (PI1**4))*SUMTWO

       FRACN = FRACNTWO - FRACNONE
       FRACTION(BANDS) = FRACTION(BANDS) + ABS(FRACN)

200	 CONTINUE

          DO 230 K=idxlo(3),idxhi(3)
            DO 220 J=idxlo(2),idxhi(2)
              DO 210 I=idxlo(1),idxhi(1)

              IF (PCELL(I,J,K).EQ.FFIELD) THEN

      ABSKG(I,J,K) = ESRCG(I,J,K)/(FRACNSUM(I,J,K)*SIG*PI1*
     &TG(I,J,K)**4)

      abskgtemp = abskg(i,j,k)
      if(abskgtemp.le.1.0d-06)abskgtemp=1.0d-06

      IF (LPATCHMEAN)THEN

       ABSKG(I,J,K) = PATCHN(I,J,K)/PATCHD(I,J,K)
       abskgtemp = abskg(i,j,k)
       if(abskgtemp.le.1.0d-06)abskgtemp = 1.0d-06
       ESRCG(I,J,K) = FRACNSUM(I,J,K)*SIG*PI1*ABSKG(I,J,K)*TG(I,J,K)**4

      END IF

c     Anisotropic scattering properties for the interior nodes
c     (for spherical harmonics only)

            scatcoef = 0.0d0
            phfunc   = 0.0d0
            shgamma(i,j,k) = 3.0d0*(abskgtemp + scatcoef) - 
     &      phfunc*scatcoef
            shgamma(i,j,k) = 1.0d0/shgamma(i,j,k)

              ENDIF

210           CONTINUE
220         CONTINUE
230       CONTINUE

	RETURN
	END






