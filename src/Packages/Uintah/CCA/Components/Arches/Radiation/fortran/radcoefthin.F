#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/radcoefthin_fort.h>
C***********************************************************************
C	SUBROUTINE RADCAL - calculates spectral gas and soot absorption  
C	coefficients for cartesian or cylindrical geometries.
C	(particle radiation properties are calculated in GLACIER)
C***********************************************************************
C      IMPLICIT NONE 
	
      double precision PI,SIG,CTWO,PI1,PTOT,SDWEAK, 
     &TEMP,ABSKP,FV,FACTOR,MUONE,MUTWO,SUMONE,SUMTWO,
     &FRACNONE,FRACNTWO,small,wavemin,wavemax,bands,dom,omega,
     &SDTOTAL

      Integer I,II,J,K,L,M

      double precision GC(5),PkPa(6),PkPaL(6)

      DATA PI/3.141593D0/,SIG/5.67E-08/,CTWO/1.4388d0/,small/1.0d-10/

      PI1 = 1.0d0 / PI

      fraction(lambda)=1.0d0

      wavemin = 50
      wavemax = 10000;
      bands = 50; 
      dom = (wavemax - wavemin)/(bands - 1); 
      omega = wavemin - dom;

        DO 200 M=1,bands

                omega = omega + dom

          DO 190 K=idxlo(3),idxhi(3)
            DO 180 J=idxlo(2),idxhi(2)
              DO 170 I=idxlo(1),idxhi(1)

              IF (PCELL(I,J,K).EQ.FFIELD) THEN

		 if (m.eq.1)ABSKG(I,J,K) = 0.0D0
		 if (m.eq.1)ESRCG(I,J,K) = 0.0D0

       		 SDTOTAL = 0.0d0              

                  DO 160 L=1,2

                 SDWEAK  = 0.0d0
                 GC(L)   = 0.0d0

c     Partial Pressures of each specie in (atm)

                 PkPa(1) = CO2(I,J,K)
                 PkPa(2) = H2O(I,J,K)
                 PkPa(3) = 0.0d0
                 PkPa(4) = 0.0d0
                 PkPa(5) = 0.0d0
                 PkPa(6) = (1.0d0-(CO2(I,J,K)+H2O(I,J,K)))

                 PkPaL(L)=(273./TG(I,J,K))*PkPa(L)*100.0d0*OPL

                 GC(L)=0.0d0
                 PTOT=0.0d0
                DO 110 II=1,6
                 PTOT=PkPa(II)+PTOT
                 GC(L)=GC(L)+RGAMMA(4*(II-1)+L)*PkPa(II)*
     &           (273.0d0/TG(I,J,K))**0.5
110             CONTINUE
                 GC(L)=GC(L)+RGAMMA(4*(7-1)+L)*PkPa(L)*273.0d0/TG(I,J,K)

111              IF(PkPa(L).EQ.0.0d0) GO TO 117

                 TEMP = TG(I,J,K)

                 GO TO(112,113),L

112              CALL ECO2(OMEGA,TEMP,GC,SDWEAK,SD15) 
                 GO TO 116

113              CALL EH2O(OMEGA,TEMP,GC,SD,PkPaL,SDWEAK)
                 GO TO 116

116              CONTINUE

c     Note: RADCAL gives mass absorption coefficients at standard density 
c     conditions(1/cm-STP).The following converts it to (1/m)

                 IF (SDWEAK.LE.small)SDWEAK = small

c     Calculate the factor for equivalent path length

      IF (L.EQ.1) FACTOR = (273.0d0*CO2(I,J,K))/TEMP
      IF (L.EQ.2) FACTOR = (273.0d0*H2O(I,J,K))/TEMP

c     Convert to (1/cm)

                 SDWEAK = SDWEAK*FACTOR

c     Convert to (1/m)

                 SDWEAK = SDWEAK*100
                 SDTOTAL = SDTOTAL+SDWEAK

117             CONTINUE

160             CONTINUE

C     Calculate blackbody emissive intensity (emissive power / pi)
C     Based on function for fractional emission in each waveband 
C     provided by Chang and Rhee (1984).

       MUONE = CTWO*OMEGA/TG(I,J,K)
       MUTWO = CTWO*(OMEGA+DOM)/TG(I,J,K)
       
       SUMONE = 0.0
       SUMTWO = 0.0

       DO 162 II=1,100

          SUMONE = SUMONE + (EXP(-II*MUONE)/II)*((MUONE**3) + 
     &(3*(MUONE**2)/II)+(6*MUONE/(II**2))+(6/(II**3)))

          SUMTWO = SUMTWO + (EXP(-II*MUTWO)/II)*((MUTWO**3) + 
     &(3*(MUTWO**2)/II)+(6*MUTWO/(II**2))+(6/(II**3)))

 162   CONTINUE

       FRACNONE = (15 * (PI1**4))*SUMONE
       FRACNTWO = (15 * (PI1**4))*SUMTWO

       FRACTION(M) = FRACNTWO - FRACNONE

       ESRCG(I,J,K) = ESRCG(I,J,K) + SDTOTAL*
     & ABS(FRACTION(M))*SIG*PI1*TG(I,J,K)**4

              ENDIF

170           CONTINUE
180         CONTINUE
190       CONTINUE

200	 CONTINUE


          DO 230 K=idxlo(3),idxhi(3)
            DO 220 J=idxlo(2),idxhi(2)
              DO 210 I=idxlo(1),idxhi(1)

              IF (PCELL(I,J,K).EQ.FFIELD) THEN

		 ABSKP = 0.0d0

                 ABSKG(I,J,K) = ESRCG(I,J,K)/(SIG*PI1*TG(I,J,K)**4)

c     Calculate Planck mean for soot in (1/cm),Pg.404(Modest, 2nd ed),C0=7.0   

		 ABSKP = 26.0d0*SFV(I,J,K)*TG(I,J,K)/CTWO

c     Convert to (1/m)

		 ABSKP = ABSKP*100

                 ABSKG(I,J,K) = ABSKG(I,J,K) + ABSKP  

              ENDIF

210           CONTINUE
220         CONTINUE
230       CONTINUE


	RETURN
	END






