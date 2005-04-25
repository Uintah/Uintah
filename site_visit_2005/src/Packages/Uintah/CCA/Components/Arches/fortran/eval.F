c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/eval_fort.h>
#else
      SUBROUTINE eval(indexLow, indexHigh, NITER, PCELL,
     &     UU, VV, WW,
     &     FF,
     &     DEN, 
     &     flowin, fmixin,
     &     CGF, ICG, IJKEVL,
     &     CRN, TAREA, XMFL, FAV,
     &     SEW, SNS, STB,
     &     RR, RA, RV,
     &     XX, XU, YY, YV, ZZ, ZW,
     &     EFAC, WFAC, NFAC, SFAC, TFAC, BFAC,
     &     YF, HL, CPDT,
     &     TEMP3,
     &     ierr)
C--------------------------------------------------------------------
C     This subroutine evaluates average mass flow rates and
C     mixture fractions at every i plane and writes them out
C--------------------------------------------------------------------

      implicit none

#include "param4.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer NITER
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision FF(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision flowin
      double precision fmixin
      double precision CGF
      integer ICG, IJKEVL
      double precision CRN
      double precision TAREA(indexLow(1):indexHigh(1))
      double precision XMFL(indexLow(1):indexHigh(1))
      double precision FAV(indexLow(1):indexHigh(1))
      double precision SEW(indexLow(1):indexHigh(1))
      double precision SNS(indexLow(2):indexHigh(2))
      double precision STB(indexLow(3):indexHigh(3))
      double precision RR(indexLow(2):indexHigh(2))
      double precision RA(indexLow(2):indexHigh(2))
      double precision RV(indexLow(2):indexHigh(2))
      double precision XX(indexLow(1):indexHigh(1))
      double precision XU(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision YV(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision ZW(indexLow(3):indexHigh(3))
      double precision EFAC(indexLow(1):indexHigh(1))
      double precision WFAC(indexLow(1):indexHigh(1))
      double precision NFAC(indexLow(2):indexHigh(2))
      double precision SFAC(indexLow(2):indexHigh(2))
      double precision TFAC(indexLow(3):indexHigh(3))
      double precision BFAC(indexLow(3):indexHigh(3))
      double precision YF
      double precision HL
      double precision CPDT
      double precision TEMP3(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer iBegActual, jBegActual, kBegActual
      integer iEndActual, jEndActual, kEndActual
      logical LENODE
      character dir
      integer i, j, k
      integer IC, JC, KC, ISTART, IEND, JSTART, JEND, KSTART, KEND
      double precision AREAEW, AREANS, AREATB
      double precision DENIJK
      double precision FAVE
      double precision FIJK
      double precision UXIJK, VXIJK

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)
      iBegActual = iBegGhost+1
      jBegActual = jBegGhost+1
      kBegActual = kBegGhost+1
      iEndActual = iEndGhost-1
      jEndActual = jEndGhost-1
      kEndActual = kEndGhost-1

c*********************************************************************
c     begin : 
c*********************************************************************
C
      OPEN (UNIT = 33,FILE = 'EVAL',STATUS = 'UNKNOWN')
      WRITE (33,999) NITER,CRN
      LENODE = .FALSE.
C     First: if integrating across i planes
      IF (IJKEVL.EQ.1) THEN
C--------------------------------------------------------------------
C calculate the average mixture fraction at every axial location
C--------------------------------------------------------------------
       JC = (jEndGhost + 1)/2
       KC = (kEndGhost + 1)/2
       JSTART = jBegActual
       JEND = jEndActual
c..lebs...
c         jstart = 38
c         jend = jEndActual
       IF (LENODE) THEN
         WRITE (33,1000) FLOWIN,FMIXIN,JC,KC
         ISTART = iBegGhost
         IEND = iEndActual
       ELSE
         WRITE (33,2000) FLOWIN,FMIXIN,JC,KC
         ISTART = iBegActual
         IEND = iEndGhost
       END IF
       DO 100 I = ISTART,IEND
	TAREA(I) = ZERO
        FAVE = ZERO
        XMFL(I) = ZERO
        DO 50 J = JSTART,JEND
         DO 25 K = kBegActual,kEndActual
          IF (PCELL(I,J,K).NE.WALL .OR. PCELL(I,J,K).NE.SYMTRY) THEN
           AREAEW = RA(J)*SNS(J)*STB(K)
C--------------------------------------------------------------------
C calculate the velocities, den, & f at the node points and faces
C--------------------------------------------------------------------
           IF (LENODE) THEN
             UXIJK = (EFAC(I)*UU(I+1,J,K) + WFAC(I)*UU(I,J,K))
             DENIJK = DEN(I,J,K)
             FIJK = FF(I,J,K)
           ELSE
             UXIJK = UU(I,J,K)
             DENIJK = PT5*(DEN(I-1,J,K)+DEN(I,J,K))
             FIJK = PT5*(FF(I-1,J,K)+FF(I,J,K))
           END IF
C--------------------------------------------------------------------
C calculate the mass flow rate and mixt. fract. at the x plane
C--------------------------------------------------------------------
           XMFL(I) = XMFL(I) + AREAEW*DENIJK*UXIJK
           FAVE = FAVE + FIJK*AREAEW*DENIJK*UXIJK
	   TAREA(I) = TAREA(I) + AREAEW
          END IF
 25      CONTINUE
 50     CONTINUE
C--------------------------------------------------------------------
C calculate centerline and average mass weighted  mixture fractions
C--------------------------------------------------------------------
        IF (XMFL(I) .NE. ZERO) THEN
          FAV(I) = ABS(FAVE/XMFL(I))
        ELSE
          FAV(I) = ZERO
        END IF
C--------------------------------------------------------------------
C output
C--------------------------------------------------------------------
        IF (LENODE) THEN
          WRITE(33,3000)I,XX(I),TAREA(I),XMFL(I),FAV(I),FF(I,JC,KC)
        ELSE
          WRITE(33,3000)I,XU(I),TAREA(I),XMFL(I),FAV(I),
     &                    PT5*(FF(I-1,JC,KC)+FF(I,JC,KC))
        END IF
  100  CONTINUE
       IF (.NOT.LENODE) WRITE (33,4000) IEND,FF(IEND,JC,KC)
C      Second: if integrating across j-planes
       ELSE IF (IJKEVL.EQ.2) THEN
C--------------------------------------------------------------------
C calculate the average mixture fraction at every axial location
C--------------------------------------------------------------------
       IC = (iEndGhost + 1)/2
       KC = (kEndGhost + 1)/2
       IF (LENODE) THEN
         WRITE (33,1005) FLOWIN,FMIXIN,IC,KC
         JSTART = jBegGhost
         JEND = jEndActual
       ELSE
         WRITE (33,2005) FLOWIN,FMIXIN,IC,KC
         JSTART = jBegActual
         JEND = jEndGhost
       END IF
       DO 200 J = JSTART,JEND
	TAREA(J) = ZERO
        FAVE = ZERO
        XMFL(J) = ZERO
        DO 150 I = iBegActual,iEndActual
         DO 125 K = kBegActual,kEndActual
          IF (PCELL(I,J,K).NE.WALL .OR. PCELL(I,J,K).NE.SYMTRY) THEN
           AREANS = RV(J)*SEW(I)*STB(K)
C--------------------------------------------------------------------
C calculate the velocities, den, & f at the node points and faces
C--------------------------------------------------------------------
           IF (LENODE) THEN
             VXIJK = (NFAC(J)*VV(I,J+1,K)+SFAC(J)*VV(I,J,K))
             DENIJK = DEN(I,J,K)
             FIJK = FF(I,J,K)
           ELSE
             VXIJK = VV(I,J,K)
             DENIJK = PT5*(DEN(I,J-1,K)+DEN(I,J,K))
             FIJK = PT5*(FF(I,J-1,K)+FF(I,J,K))
           END IF
C--------------------------------------------------------------------
C calculate the mass flow rate and mixt. fract. at the y plane
C--------------------------------------------------------------------
           XMFL(J) = XMFL(J) + AREANS*DENIJK*VXIJK
           FAVE = FAVE + FIJK*AREANS*DENIJK*VXIJK
	   TAREA(J) = TAREA(J) + AREANS
          END IF
 125     CONTINUE
 150    CONTINUE
C--------------------------------------------------------------------
C calculate centerline and average mass weighted  mixture fractions
C--------------------------------------------------------------------
        IF (XMFL(J) .NE. ZERO) THEN
          FAV(J) = ABS(FAVE/XMFL(J))
        ELSE
          FAV(J) = ZERO
        END IF
C--------------------------------------------------------------------
C output
C--------------------------------------------------------------------
        IF (LENODE) THEN
          WRITE(33,3000)J,YY(J),TAREA(J),XMFL(J),FAV(J),FF(IC,J,KC)
        ELSE
          WRITE(33,3000)J,YV(J),TAREA(J),XMFL(J),FAV(J),
     &                    PT5*(FF(IC,J-1,KC)+FF(IC,J,KC))
        END IF
  200  CONTINUE
       IF (.NOT.LENODE) WRITE (33,4000) ICG,FF(IC,ICG,KC)
C      Third: if integrating across k-planes
       ELSE IF (IJKEVL.EQ.3) THEN
C--------------------------------------------------------------------
C calculate the average mixture fraction at every axial location
C--------------------------------------------------------------------
       IC = (iEndGhost + 1)/2
       JC = (jEndGhost + 1)/2
       IF (LENODE) THEN
         WRITE (33,1010) FLOWIN,FMIXIN,IC,JC
         KSTART = kBegGhost
         KEND = kEndActual
       ELSE
         WRITE (33,2010) FLOWIN,FMIXIN,IC,JC
         KSTART = kBegActual
         KEND = kEndGhost
       END IF
       DO 300 K = KSTART,KEND
	TAREA(K) = ZERO
        FAVE = ZERO
        XMFL(K) = ZERO
        DO 250 I = iBegActual,iEndActual
         DO 225 J = jBegActual,jEndActual
          IF (PCELL(I,J,K).NE.WALL .OR. PCELL(I,J,K).NE.SYMTRY) THEN
           AREATB = SEW(I)*SNS(J)
C--------------------------------------------------------------------
C calculate the velocities, den, & f at the node points and faces
C--------------------------------------------------------------------
           IF (LENODE) THEN
             VXIJK = (TFAC(K)*WW(I,J,K+1)+BFAC(K)*WW(I,J,K))
             DENIJK = DEN(I,J,K)
             FIJK = FF(I,J,K)
           ELSE
             VXIJK = WW(I,J,K)
             DENIJK = PT5*(DEN(I,J,K-1)+DEN(I,J,K))
             FIJK = PT5*(FF(I,J,K-1)+FF(I,J,K))
           END IF
C--------------------------------------------------------------------
C calculate the mass flow rate and mixt. fract. at the z plane
C--------------------------------------------------------------------
           XMFL(K) = XMFL(K) + AREATB*DENIJK*VXIJK
           FAVE = FAVE + FIJK*AREATB*DENIJK*VXIJK
	   TAREA(K) = TAREA(K) + AREANS
          END IF
 225     CONTINUE
 250    CONTINUE
C--------------------------------------------------------------------
C calculate centerline and average mass weighted  mixture fractions
C--------------------------------------------------------------------
        IF (XMFL(K) .NE. ZERO) THEN
          FAV(K) = ABS(FAVE/XMFL(K))
        ELSE
          FAV(K) = ZERO
        END IF
C--------------------------------------------------------------------
C output
C--------------------------------------------------------------------
        IF (LENODE) THEN
          WRITE(33,3000)K,ZZ(K),TAREA(K),XMFL(K),FAV(K),FF(IC,JC,K)
        ELSE
          WRITE(33,3000)K,ZW(K),TAREA(K),XMFL(K),FAV(K),
     &                    PT5*(FF(IC,JC,K-1)+FF(IC,JC,K))
        END IF
  300  CONTINUE
       IF (.NOT.LENODE) WRITE (33,4000) ICG,FF(IC,JC,ICG)
       END IF
C
C-----------------------------------------------------------------------
C     INTEGRATED OUTLET STUFF
C-----------------------------------------------------------------------
C     Convert to molar parts per million for final output.
C     Store all results for output in TEMP arrays.
C     Also get a local (then an integrated) value for thermal content
      DO 1220 K = kBegGhost,kEndGhost
        DO 1210 J = jBegGhost,jEndGhost
          DO 1200 I = iBegGhost,iEndGhost
            IF (PCELL(I,J,K) .NE. WALL) THEN
              YF = FF(I,J,K)
              TEMP3(I,J,K) = (ONE-HL)*CPDT
            END IF
 1200     CONTINUE
 1210   CONTINUE
 1220 CONTINUE
      IF (IJKEVL.EQ.1) DIR = 'I'
      IF (IJKEVL.EQ.2) DIR = 'J'
      IF (IJKEVL.EQ.3) DIR = 'K'
C     Print at ICG
      WRITE(33,*)'       '
C-------------------------------------------------------------------------
C    Get BLOCK correction stuff for f
C-------------------------------------------------------------------------
      IF (CGF.GT.ZERO) THEN
        CGF = ONE
        IF (FAV(ICG).GT.ZERO) CGF = FMIXIN/FAV(ICG)
        WRITE(33,7000) (CGF-ONE)*CENT,ICG
      END IF
C-------------------------------------------------------------------------
       CALL FLUSH(33)
       CLOSE (UNIT=33)
C
      RETURN
  999  FORMAT(//25X,'at iteration # ',I5/
     &          25X,' avg. cell Re  ',1PE10.3)
 1000  FORMAT(/2X,'i',4X,'x (node)',4X,'x-section',2X,
     & 'mass flow rate',3X,'mix. fract.',2X,'centerline f'/
     & 32X,'(',1PE10.3,')',3X,'(',1PE10.3,')',3X,'(',i3,1x,i3,')',
     & 3X,'(',1PE10.3,')')
 1005  FORMAT(/2X,'j',4X,'y (node)',4X,'x-section',2X,
     & 'mass flow rate',3X,'mix. fract.',2X,'centerline f'/
     & 32X,'(',1PE10.3,')',3X,'(',1PE10.3,')',3X,'(',i3,1x,i3,')',
     & 3X,'(',1PE10.3,')')
 1010  FORMAT(/2X,'k',4X,'z (node)',4X,'x-section',2X,
     & 'mass flow rate',3X,'mix. fract.',2X,'centerline f'/
     & 32X,'(',1PE10.3,')',3X,'(',1PE10.3,')',3X,'(',i3,1x,i3,')',
     & 3X,'(',1PE10.3,')')
 2000  FORMAT(//2X,'i',4X,'x (face)',4X,'x-section',2X,
     & 'mass flow rate',3X,'mix. fract.',2X,'centerline f'/
     & 32X,'(',1PE10.3,')',3X,'(',1PE10.3,')',3X,'(',i3,1x,i3,')',
     & 3X,'(',1PE10.3,')')
 2005  FORMAT(//2X,'j',4X,'y (face)',4X,'x-section',2X,
     & 'mass flow rate',3X,'mix. fract.',2X,'centerline f'/
     & 32X,'(',1PE10.3,')',3X,'(',1PE10.3,')',3X,'(',i3,1x,i3,')',
     & 3X,'(',1PE10.3,')')
 2010  FORMAT(//2X,'k',4X,'z (face)',4X,'x-section',2X,
     & 'mass flow rate',3X,'mix. fract.',2X,'centerline f'/
     & 32X,'(',1PE10.3,')',3X,'(',1PE10.3,')',3X,'(',i3,1x,i3,')',
     & 3X,'(',1PE10.3,')')
 3000  FORMAT(I3,3X,1PE10.3,3X,1PE10.3,4X,1PE10.3,5X,1PE10.3,3X,
     &        1PE10.3,4X,1PE10.3)
 4000  FORMAT(I3,'  (node point)',44X,1PE10.3)
 5000  FORMAT(I3,'  NODE:  Integrated CO (wet) ',1PE10.3,' ppm,  ',
     & ' Integrated CO (dry) ',1PE10.3,' ppm')
 6000  FORMAT(I3,'  NODE:  Integrated O2 (wet) ',1PE10.3,' %,    ',
     & ' Integrated O2 (dry) ',1PE10.3,' %')
 7000  FORMAT(/' Correcting next guess for all fs by: ',1PE10.3,
     & ' % based on integral of f at plane ',I3)
 8000  FORMAT(/' ENERGY BALANCE: ( outlet at plane =', I4,')',
     & /15X,'      Adiabatic Outlet Temp: ',4X,F6.1,' K',
     & /15X,'Mixed & Reacted Outlet Temp: ',4X,F6.1,' K',
     & ' (with avg. computed hl)',
     & /15X,'     Integrated Outlet Temp: ',4X,F6.1,' K',
     & /15X,'Integrated Outlet Heat Loss: ',4X,F6.2,' %',
     & /15X,
     & /15X,' Approx. Gas Thermal Output: ',1PE10.3,
     & ' watts (at current burnout)',
     & /15X,'     Gaseous Radiation Loss: ',1PE10.3,' watts',
     & /15X,'    Particle Radiation Loss: ',1PE10.3,' watts',
     & /15X,'             PV Energy Loss: ',1PE10.3,' watts',
     & /15X,'         Prtl-Gas Energy Ex: ',1PE10.3,' watts',
     & /15X,'Captrd Particle Energy Loss: ',1PE10.3,
     &' watts (after last p iter)',
     & /15X,'Outflw Particle Energy Loss: ',1PE10.3,
     &' watts (after last p iter)',
     & /15X,'            Convective Loss: ',1PE10.3,' watts',
     & /15X,' Inlet stream Thermal Input: ',1PE10.3,' watts',
     & /15X,'  Approximate Thermal Input: ',1PE10.3,
     & ' watts (at current burnout)',
     & /15X,
     & /15X,'Total Balance Percent Error: ',0PF6.1,
     &' % (after last p iter)',
     & /15X,'  Gas Balance Percent Error: ',0PF6.1,' %',
     & /15X,'Partl Balance Percent Error: ',0PF6.1,
     &' % (after last p iter)')
 9000  FORMAT(/'  COMPOSITIONAL INFO. at pref: (',I3,',',I3,',',I3,')',
     & /15X,'                    Density: ',1PE10.3,' kg/m^3',
     & /15X,'                Temperature: ',4X,0PF6.1,' K',
     &50(/15X,'     mole fraction ',2A4,': ',0PF6.2,' %'))
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:45  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
