#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/radcal_fort.h>
C***********************************************************************
C	SUBROUTINE RADCAL - calculates spectral gas and soot absorption  
C	coefficients for cartesian or cylindrical geometries.
C	(particle radiation properties are calculated in GLACIER)
C***********************************************************************
C      IMPLICIT NONE 
	
      double precision PI,SIG,CTWO,PI1,PTOT,SDWEAK,UK, 
     &x,y,z,xxx,yyy,zzz,TEMP,ABSKP,FV,FACTOR,
     &MUONE,MUTWO,SUMONE,SUMTWO,FRACNONE,FRACNTWO,small

      Integer I,II,J,K,L

      double precision DELTAX(idxlo(1)-1:idxhi(1)+1),
     &GC(5),PkPa(6),PkPaL(6)

      DATA PI/3.141593D0/,SIG/5.67E-08/,CTWO/1.4388d0/,small/1.0d-10/

      PI1 = 1.0d0 / PI

      omega = omega + dom

      DO 80 K=idxlo(3),idxhi(3)
        DO 70 J=idxlo(2),idxhi(2)
          DO 60 I=idxlo(1),idxhi(1)

          IF (PCELL(I,J,K).EQ.FFIELD) THEN
            ABSKG(I,J,K)= 0.0d0
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

          DO 190 K=idxlo(3),idxhi(3)
            DO 180 J=idxlo(2),idxhi(2)
              DO 170 I=idxlo(1),idxhi(1)

              IF (PCELL(I,J,K).EQ.FFIELD) THEN

                ABSKP = 0.0D0
                   IF (LPROBONE) GO TO 163

                DO 160 L=1,4

                 SDWEAK  = 0.0d0
                 GC(L)   = 0.0d0

c     Partial Pressures of each specie in (atm)

                 PkPa(1) = CO2(I,J,K)
                 PkPa(2) = H2O(I,J,K)
                 PkPa(3) = 0.0d0
                 PkPa(4) = 0.0d0
                 PkPa(5) = 0.0d0
c                 PkPa(6) = 0.0d0
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

                 GO TO(112,113,114,115),L

112              CALL ECO2(OMEGA,TEMP,GC,SDWEAK,SD15)     
                 GO TO 116
113              CALL EH2O(OMEGA,TEMP,GC,SD,PkPaL,SDWEAK)
                 GO TO 116
c114              CALL EFUEL(OMEGA,TEMP,PkPa,PTOT,GC,SDWEAK,SD7,SD3)
114              CONTINUE
                 GO TO 116
c115              CALL ECO(OMEGA,TEMP,SDWEAK)
115              CONTINUE

116              CONTINUE

c     Note: RADCAL gives mass absorption coefficients at standard density 
c     conditions(1/cm-STP).The following converts it to (1/m)

                 IF (SDWEAK.LE.small)SDWEAK = small

c     Calculate the factor for equivalent path length
      IF (L.EQ.1) FACTOR = (273.0d0*CO2(I,J,K))/TEMP
      IF (L.EQ.2) FACTOR = (273.0d0*H2O(I,J,K))/TEMP
c      IF (L.EQ.4) FACTOR = (273.0d0*0.03)/TEMP

c     Convert to (1/cm)
                 SDWEAK = SDWEAK*FACTOR

c     Convert to (1/m)

                 SDWEAK = SDWEAK*100
                 ABSKG(I,J,K)= ABSKG(I,J,K)+SDWEAK

117              continue

160             CONTINUE

                 FV = SFV(I,J,K)
                 CALL SOOT(OMEGA,FV,ABSKP)

                 ABSKG(I,J,K)=ABSKG(I,J,K)+ABSKP
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

       FRACTION(bands) = FRACNTWO - FRACNONE

       ESRCG(I,J,K) = ABS(FRACTION(bands))*SIG*PI1*ABSKG(I,J,K)* 
     & TG(I,J,K)**4

 163                CONTINUE

       IF (LPROBONE) THEN
                 x=0.5-xx(I)
                 y=0.5-yy(J)
                 z=0.5-zz(K)
                 if (xx(I).ge.0.5)
     &           x=xx(I)-0.5
                 if (yy(J).ge.0.5)
     &           y=yy(J)-0.5
                 if (zz(K).ge.0.5)
     &           z=zz(K)-0.5
                 xxx=x/0.5
                 yyy=y/0.5
                 zzz=z/0.5

                 ABSKG(I,J,K)=(0.9*(1-xxx)*(1-yyy)*(1-zzz))+0.1
		 TG(I,J,K)=(1/SIG)**0.25

                 MUONE = CTWO*OMEGA/TG(I,J,K)
                 MUTWO = CTWO*(OMEGA+DOM)/TG(I,J,K)

                 SUMONE = 0.0
                 SUMTWO = 0.0

              DO 165 II=1,100

          SUMONE = SUMONE + (EXP(-II*MUONE)/II)*((MUONE**3) + 
     &(3*(MUONE**2)/II)+(6*MUONE/(II**2))+(6/(II**3)))

          SUMTWO = SUMTWO + (EXP(-II*MUTWO)/II)*((MUTWO**3) + 
     &(3*(MUTWO**2)/II)+(6*MUTWO/(II**2))+(6/(II**3)))

 165          CONTINUE

              FRACNONE = (15 * (PI1**4))*SUMONE
              FRACNTWO = (15 * (PI1**4))*SUMTWO

              FRACTION(bands) = FRACNTWO - FRACNONE

              ESRCG(I,J,K) = ABS(FRACTION(bands))*PI1 * ABSKG(I,J,K)

      ENDIF

              ENDIF

170           CONTINUE
180         CONTINUE
190       CONTINUE

	RETURN
	END






