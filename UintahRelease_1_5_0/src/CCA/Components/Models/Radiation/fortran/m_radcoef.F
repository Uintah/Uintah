C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
C 
#include <CCA/Components/Models/Radiation/fortran/m_radcoef_fort.h>
C***********************************************************************
C	SUBROUTINE RADCOEF - calculates gas and soot absorption  
C	coefficients for cartesian or cylindrical geometries.
C***********************************************************************

c       Ported to models infrastructure by
c       Seshadri Kumar, April 11, 2005
c       Commented out pcell .eq. ffield stuff, because that is always
c       true with Gautham's test problems, which is all that logical 
c       test is ever used for.

      double precision PI,PI1,PATM,OPF,EMG,PCO2,PH2O,PCO2F,PH2OF,
     &PSUMF,PAVE,PML,PRES,PWPC,TGI,CCO2,CH2O,DELE,EMGI,EMGF,FV,ECO2,
     &EH2O,TGF,SIG,xxx,yyy,zzz,xxxx,yyyy,zzzz,small,QSRC,M,scatcoef, 
     &phfunc,abskgtemp

      double precision AEM(3)
      double precision AEK(3)

      Integer I,II,J,K,NELEM

      DATA SIG/5.67D-08/, small/1.0d-40/
C
      fraction(lambda)=1.0d0
      fractiontwo(lambda)=1.0d0

      pi = dacos(-1.0D0)

      PI1 = 1.0d0 / PI
      PRES = 1.01325D5
      PATM = PRES / 1.01325D5

c     Conversion of OPL in "meters" to "feet"
      OPF = 3.281d0 * opl

c     Domain details for Problem Two

      IF (LPROBTWO)THEN

        DO 90 K=idxlo(3),idxhi(3)
          DO 85 J=idxlo(2),idxhi(2)
            DO 80 I=idxlo(1),idxhi(1)

c              IF (PCELL(I,J,K).EQ.FFIELD) THEN

	       CO2(I,J,K) = 0.4d0*xx(I)*(1.0d0-xx(I))+0.06d0
	       H2O(I,J,K) = 2.0d0*CO2(I,J,K)
	       SFV(I,J,K) = (40.0d0*xx(I)*(1.0d0-xx(I))+6.0d0)*1.0d-07
	       TG(I,J,K) = 4000.0d0*xx(I)*(1.0d0-xx(I))+800.0d0
                  
c              END IF

 80	    CONTINUE
 85	 CONTINUE
 90	CONTINUE

        END IF

        IF (LRADCAL)GO TO 441
        IF (LPROBONE.OR.LPROBTHREE)GO TO 241

C       Calculate absorption coefficients for FFIELD's

	DO 240 K=idxlo(3),idxhi(3)
	   DO 230 J=idxlo(2),idxhi(2)
              DO 220 I=idxlo(1),idxhi(1)
         
c		 IF (PCELL(I,J,K).EQ.FFIELD) THEN
		    EMG = 0.0d0
C-----------------------------------------------------------------------
C       This section calculates CO2-H2O mixture gas emissivities based
C       on a combination of a Hottel chart curve fit (originally done
C       by Phil Smith and students) and a curve fit of high temperature
C       exponential wide band spectral predictions by Coppalle and 
C       Vervisch (Combustion and Flame, Vol 49, pp. 101-108, 1983).
C       If the gas temperature is less than 1200 K, work with PJS model;
C       if the temperature is greater than 2100 K work with C & V model;  
C       if the temperature is 1200-2100 K wotk with linear interpolation
C       between the Hottel chart point at 1200 K and the C & V point
C       at 2100 K. This makes the routine run a little slower, but it
C       is still much faster than Modak's model and just as accurate 
C       (over a wider temperature range, i.e., 300-3000 K).
C
C       Note: Since most temperatures will be 1200-2100 K, do both
C             models for all temperatures and determine appropriate
C             emissivity at end. This improves run time by
C             eliminating a nested IF statement. Also, run times can be
C             cut almost in half if only one IF option is used with the
C             C & V model, e.g., only Pw/Pc = 1 or TG < 2500 K is used.
C----------------------------------------------------------------------
                  PCO2 = CO2(I,J,K) * PATM
                  PH2O = H2O(I,J,K) * PATM

                  PCO2F = PCO2 * OPF
                  PH2OF = PH2O * OPF
                  PSUMF = MAX((PCO2F + PH2OF),0.0001d0)
                  PAVE = (PATM + PH2O) * 0.5d0
                  PML = PSUMF / 3.281d0
                  PWPC = PH2O / (PCO2+small)
                  IF (PWPC .LE. 1.5d0) THEN
                    AEK(1) = 0.464d0 * PML
                    AEK(2) = 3.47d0 * PML
                    AEK(3) = 121.6d0 * PML
                  ELSE
                    AEK(1) = 0.527d0 * PML
                    AEK(2) = 3.78d0 * PML
                    AEK(3) = 99.54d0 * PML
                  END IF
C
C       EICO2 and EIH2O exploit Hottel charts to calculate (curve-fit)
C       emissivities for carbon dioxide and water vapor, respectively;
C       PLA units are atm-ft, Pc and Pw are in atm, TGI is in K
C
                  TGI = MIN(TG(I,J,K),1200.0d0)
                  CALL m_EICO2( TGI,PCO2F,ECO2 )
                  CALL m_EIH2O( TGI,PH2OF,EH2O )

C       Calculate total gas emissivity accounting for spectral overlap
C       and pressure correction
C
                  CCO2 = 1.0d0
                  CH2O = 1.0d0
                  IF (PCO2F .GE. 0.5d0) THEN
                    CCO2 = 1.0d0 + 0.14427d0 * DLOG(PATM)
                  ELSE IF (PCO2F .GT. 0.0d0) THEN
                    CCO2 = 1.0d0 + 0.24663d0 * DLOG(PATM)
                  END IF
                  IF (PH2OF .GT. 1.0d0) THEN
                    CH2O = 1.29703d0 + 0.43168d0 * DLOG(PAVE)
                  ELSE IF (PH2OF .GT. 0.0d0) THEN
                    CH2O = 1.46701d0 + 0.6663d0 * DLOG(PAVE)
                  END IF
                  IF (TGI .GE. 1005.0d0) THEN
                    DELE = 0.03047d0 + 0.01717d0 * DLOG(PSUMF)
                    DELE = MIN(DELE,0.06d0)
                    DELE = MAX(DELE,0.0d0)
                  ELSE
                    DELE = 0.01544d0 + 0.01045d0 * DLOG(PSUMF)
                    DELE = MIN(DELE,0.035d0)
                    DELE = MAX(DELE,0.0d0)
                  END IF
                  EMGI = CCO2 * ECO2 + CH2O * EH2O - DELE

C       Work with Coppalle and Vervisch model to calculate CO2-H2O mixture 
C       total emissivity by summing 3 gray gases (plus one clear).
C       Compute emissivity weighting factors for CO2-H2O mixture
C       based on linear fit of temperature (for Ptot = 1 atm, TGF in K).
C       First coefficient set is for Pw/Pc = 1, second is for Pw/Pc = 2.
C
                  TGF = MAX(TG(I,J,K),2100.0d0)
                  IF (TGF .LT. 2500.0d0) THEN
                    AEM(1) = 0.136d0 + 0.0000726d0 * TGF
                    AEM(2) = 0.516d0 - 0.000163d0 * TGF
                    AEM(3) = 0.0517d0 - 0.0000176d0 * TGF
                  ELSE
                    AEM(1) = 0.464d0 - 0.0000596d0 * TGF
                    AEM(2) = 0.336d0 - 0.0000909d0 * TGF
                    AEM(3) = 0.0245d0 - 0.00000654d0 * TGF
                  END IF
                  IF (TGF.LT.2500.0d0 .AND. PWPC.GE.1.5d0) THEN
                    AEM(1) = 0.132d0 + 0.0000725d0 * TGF
                    AEM(2) = 0.547d0 - 0.000171d0 * TGF
                    AEM(3) = 0.0489d0 - 0.0000176d0 * TGF
                  ELSE IF (PWPC .GE. 1.5d0) THEN
                    AEM(1) = 0.430d0 - 0.0000472d0 * TGF
                    AEM(2) = 0.370d0 - 0.000101d0 * TGF
                    AEM(3) = 0.0184d0 - 0.00000511d0 * TGF
                  END IF
                  EMGF = AEM(1) * (1.0d0 - EXP(-AEK(1)))
     1                 + AEM(2) * (1.0d0 - EXP(-AEK(2)))
     2                 + AEM(3) * (1.0d0 - EXP(-AEK(3)))

C       Determine which model is appropriate for given gas temperature
C
                  IF (TG(I,J,K) .LE. 1200.0d0) THEN
                    EMG = EMGI
                  ELSE IF (TG(I,J,K) .GE. 2100.0d0) THEN
                    EMG = EMGF
                  ELSE
                    EMG = EMGI + (EMGF-EMGI) 
     1                  * ((TG(I,J,K)-1200.0d0) / 900.0d0)
                  END IF

c-- first guess for SO2 effects, assume same as CO2
c                  IF (IDSO2.NE.0) THEN
c                    PSO2F = SPECIE(I,J,K,IDSO2) * PATM * OPF
c                    TSO2 = TG(I,J,K)
c                    CALL m_EICO2( TSO2,PSO2F,ESO2 )
c                    EMG = EMG + ESO2
c                  END IF

c WARNING! The second of the following limits is needed to make sure
c negative argument will not go in log function. Gautham needs to investigate
c why emg becomes greater then one.
                   EMG = MAX (EMG, 0.0d0)
                   EMG = MIN (EMG, 0.9d0)
C
C       Calculate absorption coefficient from Bouguer's Law
C

                  ABSKG(I,J,K) = -DLOG (1.0d0 - EMG) / OPL

C-----------------------------------------------------------------------
C       Soot model is based on a correlation given by Sarofim and Hottel
C       in "Radiative Transfer in Combustion Chambers: Influence of 
C       Alternative Fuels," Sixth Intl. Heat Transfer Conf., Toronto,
C       Vol 6, pp 199-217, August, 1978.  Soot formation is based on
C       mixture fraction, density and moles of carbon at each local 
C       cell. Soot destruction is based on local equivalence ratio.
C       Scattering effects due to soot agglomerates are not considered
C       (only soot absorption/emission). 
C       THIS MODEL IS NOT YET DESIGNED TO BE USED WITH PARTICLE
C       (COAL) CALCULATIONS.
C-----------------------------------------------------------------------

                    FV = SFV(I,J,K)
                    ABSKG(I,J,K) = ABSKG(I,J,K) + 
     $                   MIN( 25.0d0,((4.0d0/OPL) 
     1                   *DLOG(1.0d0 + 350.0d0*FV*TG(I,J,K)*OPL)))

c hardcoded abskg to one if not flow field
                    if (pcell(i,j,k).ne.ffield) abskg(i,j,k)=1.0d0
                    abskgtemp = abskg(i,j,k)
                    if(abskgtemp.le.1.0d-06)abskgtemp = 1.0d-06

c     Anisotropic scattering properties for the interior nodes
c     (for spherical harmonics only)

                    scatcoef = 0.0d0
                    phfunc   = 0.0d0
                    shgamma(i,j,k) = 3.0d0*
     $                   (abskgtemp + scatcoef) - 
     &                   phfunc*scatcoef
                    shgamma(i,j,k) = 1.0d0/shgamma(i,j,k)
            
C       Calculate blackbody emissive intensity (emissive power / pi)
C
                    ESRCG(I,J,K) = abs(fraction(lambda)) * 
     $                   SIG * PI1 * 
     1                   ABSKG(I,J,K) * TG(I,J,K)**4

c                END IF
220           CONTINUE
230         CONTINUE
240       CONTINUE

C-----------------------------------------------------------------------
C--- PARTICLE PROPS WILL BE CALCULATED IN GLACIER
C-----------------------------------------------------------------------
c     User specified properties for validation

241   CONTINUE

        IF (LPROBONE)THEN

          DO 340 K=idxlo(3),idxhi(3)
            DO 330 J=idxlo(2),idxhi(2)
              DO 320 I=idxlo(1),idxhi(1)

c                IF (PCELL(I,J,K).EQ.FFIELD) THEN

                 xxx=0.5-xx(I)
                 yyy=0.5-yy(J)
                 zzz=0.5-zz(K)
                 if (xx(I).ge.0.5)
     &           xxx=xx(I)-0.5
                 if (yy(J).ge.0.5)
     &           yyy=yy(J)-0.5
                 if (zz(K).ge.0.5)
     &           zzz=zz(K)-0.5
                 xxxx=xxx/0.5
                 yyyy=yyy/0.5
                 zzzz=zzz/0.5

                 ABSKG(I,J,K)=(0.9*(1-xxxx)*(1-yyyy)*(1-zzzz))+0.1
                 ESRCG(I,J,K) = abs(fraction(lambda))*PI1 * 
     &           ABSKG(I,J,K)

c     Anisotropic scattering properties for the interior nodes
c     (for spherical harmonics only)

                 scatcoef = 0.0d0
                 phfunc   = 0.0d0
                 shgamma(i,j,k) = 3.0d0*(abskg(i,j,k) + 
     $                scatcoef) - phfunc*scatcoef
                 if(shgamma(i,j,k).eq.0.0d0) 
     $                shgamma(i,j,k) = 1.0d-10
                 shgamma(i,j,k) = 1.0d0/shgamma(i,j,k)

c                END IF

320           CONTINUE
330         CONTINUE
340       CONTINUE

        END IF

        IF (LPROBTHREE)THEN

          DO 440 K=idxlo(3),idxhi(3)
            DO 430 J=idxlo(2),idxhi(2)
              DO 420 I=idxlo(1),idxhi(1)

                ABSKG(I,J,K)=2.0d0
                TG(I,J,K)=295.0d0

                ESRCG(I,J,K) = abs(fraction(lambda)) 
     &          * SIG * PI1 * ABSKG(I,J,K) * TG(I,J,K)**4

c               IF((I.EQ.12).AND.(J.EQ.12).AND.(K.EQ.12))
c     &          ESRCG(I,J,K) = 1.0/(4.0*PI*((0.64/25)**3))

420           CONTINUE
430         CONTINUE
440       CONTINUE

c       OPEN(UNIT=15,FILE='qpts.methanol.24x24x50',STATUS='OLD')
        OPEN(UNIT=15,FILE='qpts.methanol.64x64x128.1',STATUS='OLD')
        READ(15,*)NELEM

        DO II=1,NELEM
           READ(15,*)QSRC,K,J,I
           write(6,*)II
           ESRCG(I,J,K) = (1.0/(4.0*PI))*QSRC
        ENDDO

        CLOSE (UNIT=15)

        END IF

441   CONTINUE

        RETURN
        END
