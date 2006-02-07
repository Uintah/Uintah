c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/erchek_fort.h>
#else
      SUBROUTINE erchek(indexLow, indexHigh, PCELL,
     &     UU, VV, WW,
     &     XX, YY, ZZ, XU, YV, ZW,
     &     ICHEK, IPREF, JPREF, KPREF,
     &     ICG, IJKEVL,
     &     VISCOS, PRES,
     &     DIACH, DIAPR, DIASC, CHLGH, CHHGH, CHWID,
     &     FLOWPR, DENPR, TINPR, PRLS, SWNPR, TPR, FPR, ETAPR,
     &     FLOWSC, DENSC, TINSC, SCLS, SWNSC, TSC, FSC, ETASC,
     &     IJKPRT, ISKIP, JSKIP, KSKIP,
     &     INCELL,
     &     LRDGD, LPRDJK, LRDPR,
     &     LDDI1, LDDJ1, LDDK1, LDDNI, LDDNJ, LDDNK,
     &     LRSRT, LKETM, LFIXM, LNLKM, LMLTM, LPRIN,
     &     ierr)
C--------------------------------------------------------------------
C     This subroutine provides warning to users for common errors
C          confronted at the beginning and end of execution.
C--------------------------------------------------------------------

      implicit none

#include "param4.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer pcell(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision UU(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1), 
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision XX(indexLow(1):indexHigh(1))
      double precision XU(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision YV(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      double precision ZW(indexLow(3):indexHigh(3))
      integer ICHEK, IPREF, JPREF, KPREF
      integer ICG, IJKEVL
      double precision VISCOS, PRES
      double precision DIACH, DIAPR, DIASC, CHLGH, CHHGH, CHWID
      double precision FLOWPR, DENPR, TINPR, PRLS, SWNPR, TPR,FPR,ETAPR
      double precision FLOWSC, DENSC, TINSC, SCLS, SWNSC, TSC,FSC,ETASC
      integer IJKPRT, ISKIP, JSKIP, KSKIP
      integer INCELL(6, indexLow(1):indexHigh(1), 
     &     indexLow(1):indexHigh(1))
      logical LRDGD, LPRDJK, LRDPR,
     &     LDDI1, LDDJ1, LDDK1, LDDNI, LDDNJ, LDDNK,
     &     LRSRT, LKETM, LFIXM, LNLKM, LMLTM, LPRIN
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      logical LTOFAS, LOUTRC
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k, JJ, KK
      integer NMAX
      integer NX, NY, NZ, NI, NJ, NK, NI0, NJ0, NK0, NI2, NJ2, NK2
      integer NI1, NJ1, NK1, NIM1, NJM1, NKM1
      double precision FLOWT
      double precision SOFS, COMVEL

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)
      IST = iBegGhost+1
      JST = jBegGhost+1
      KST = kBegGhost+1
      IEND = iEndGhost-1
      JEND = jEndGhost-1
      KEND = kEndGhost-1

      NX = iEndGhost
      NY = jEndGhost
      NZ = kEndGhost
      NI = NX
      NJ = NY
      NK = NZ
      NI1 = iBegGhost
      NJ1 = jBegGhost
      NK1 = kBegGhost
      NI0 = NI1-1
      NJ0 = NJ1-1
      NK0 = NK1-1
      NI2 = NI1+1
      NJ2 = NJ1+1
      NK2 = NK1+1
      NIM1 = NI-1
      NJM1 = NJ-1
      NKM1 = NK-1

c*********************************************************************
c     Start computation
c*********************************************************************
      LTOFAS = .FALSE.
      LOUTRC = .FALSE.
      IF (ICHEK .EQ. 1 ) THEN
C-------------------------------------------------------------------------
C     Initial Error Checks
C
C     Input Parameter Combatablity Checks
c  
c     ierr = 101 : 'NX must be largest dimension'
c     ierr = 102 : 'No CHLGH Specified'
c     ierr = 103 : 'No CHLGH, CHWID or DIACH Specified'
c     ierr = 104 : 'No NI Specified'
c     ierr = 105 : 'No NJ Specified'
c     ierr = 106 : 'No NK Specified'
c     ierr = 107 : 'NJ not NK but periodic (ie. LPRDJK=.T.)'
c     ierr = 108 : 'No flowrates specified'
c     ierr = 109 : 'Error: No. of I cells > dimensions'
c     ierr = 110 : 'Error: No. of J cells > dimensions'
c     ierr = 111 : 'Error: No. of K cells > dimensions'
c     ierr = 112 : 'Error: NUMBER OF X LESS THAN 3 '
C-------------------------------------------------------------------------
        IF (MAX(NY,NZ).GT.NX) THEN
           ierr = 101
           RETURN
        ENDIF
        IF (.NOT.LRDGD) THEN
          IF (CHLGH.le.tiny) then
             ierr = 102
             return
          endif
          IF ((CHHGH.le.tiny).AND.(CHWID.le.tiny).AND.(DIACH.le.tiny))
     &         then
             ierr = 103
             return
          endif
          IF (NI.EQ.0) then
             ierr = 104
             return
          endif
          IF (NJ.EQ.0) then
             ierr = 105
             return
          endif
          IF (NK.EQ.0) then
             ierr = 106
             return
          endif
        END IF
        IF (LPRDJK.AND.(NJ.NE.NK)) then
           ierr = 107
           return
        END IF
        FLOWT = FLOWPR + FLOWSC
c
c     debug: comment below (for pure diffusion)
c     Will need to uncomment for general cases.
c     SK: uncommented, 04/23/95
c
        IF (FLOWT.le.tiny .AND. .NOT.LRDPR) THEN
          IF (.NOT.(LDDI1.OR.LDDJ1.OR.LDDK1.OR.LDDNI.OR.LDDNJ.OR.LDDNK))
     &          then
             ierr = 108
             return
          endif
        END IF
C--------------------------------------------------------------------
C     SIMPLE CHECKS
C--------------------------------------------------------------------
        IF (NI.GT.NX) then
           ierr = 109
           return
        endif
        IF (NJ.GT.NY) then
           ierr = 110
           return
        endif
        IF (NK.GT.NZ) then
           ierr = 111
           return
        endif
        IF (NX.LE.3) then
           ierr = 112
           return
        endif
c
c   Comment out write statements (BB)
c
        IF (LRSRT) WRITE(6,20000)
        IF (LFIXM) WRITE(6,21000)
        IF (.NOT.(IJKEVL.EQ.1.OR.IJKEVL.EQ.2.OR.IJKEVL.EQ.3)) IJKEVL = 1
        IF (IJKEVL.EQ.1.AND.(ICG.LE.1.OR.ICG.GE.NI)) ICG = NIM1
        IF (IJKEVL.EQ.2.AND.(ICG.LE.1.OR.ICG.GE.NJ)) ICG = NJM1
        IF (IJKEVL.EQ.3.AND.(ICG.LE.1.OR.ICG.GE.NK)) ICG = NKM1
        IF (LNLKM) THEN
          LKETM = .TRUE.
          WRITE (6,*) '           Turbulence Model:  non-linear k-e'
        ELSE IF (LKETM) THEN
          WRITE (6,*) '           Turbulence Model:  standard  k-e'
        END IF
        IF (LMLTM) THEN
          WRITE (6,*) '           Turbulence Model:  mixing length'
        END IF
C-----------------------------------------------------------------------
C     PRESSURE REFERENCE LOCATION CALCULATION
C-----------------------------------------------------------------------
        IF ((IPREF.LE.NI0 .OR. JPREF.LE.NI0 .OR. KPREF.LE.NI0) .OR. 
     &      (IPREF.GT.NI.OR. JPREF.GT.NJ.OR. KPREF.GT.NK)) THEN
        DO 110 K = NK1,NK
          DO 100 J = NJ1,NJ
            IF ((PCELL(NI1,J,K).NE.WALL.AND.PCELL(NI2,J,K).NE.WALL).AND.
     &          INCELL(NI1,J,K).LE.NI1) THEN
              IPREF = NI2
              JPREF = J
              KPREF = K
              GO TO 160
            END IF
            IF ((PCELL(NI,J,K).NE.WALL.AND.PCELL(NIM1,J,K).NE.WALL) 
     &           .AND. (INCELL(2,J,K).GE.NI.OR.INCELL(2,J,K).EQ.0)) 
     &           THEN
              IPREF = NIM1
              JPREF = J
              KPREF = K
              GO TO 160
            END IF
  100     CONTINUE
  110   CONTINUE
        DO 130 K = NK1,NK
          DO 120 I = NI1,NI
            IF ((PCELL(I,NJ1,K).NE.WALL.AND.PCELL(I,NJ2,K).NE.WALL).AND.
     &          INCELL(NI2+1,I,K).LE.NJ1) THEN
              IPREF = I
              JPREF = NJ2
              KPREF = K
              GO TO 160
            END IF
            IF ((PCELL(I,NJ,K).NE.WALL.AND.PCELL(I,NJM1,K).NE.WALL) 
     &         .AND. (INCELL(4,I,K).GE.NJ.OR.INCELL(4,I,K).EQ.0)) THEN
              IPREF = I
              JPREF = NJM1
              KPREF = K
              GO TO 160
            END IF
  120     CONTINUE
  130   CONTINUE
        DO 150 J = NJ1,NJ
          DO 140 I = NI1,NI
            IF ((PCELL(I,J,NK1).NE.WALL.AND.PCELL(I,J,NK2).NE.WALL).AND.
     &          INCELL(5,I,J).LE.NK1) THEN
              IPREF = I
              JPREF = J
              KPREF = NK2
              GO TO 160
            END IF
            IF ((PCELL(I,J,NK).NE.WALL.AND.PCELL(I,J,NKM1).NE.WALL) 
     &         .AND. (INCELL(6,I,J).GE.NK.OR.INCELL(6,I,J).EQ.0)) THEN
              IPREF = I
              JPREF = J
              KPREF = NKM1
              GO TO 160
            END IF
  140     CONTINUE
  150   CONTINUE
  160   CONTINUE
c-- fix pressure ref for asarco cyclone case ------
c	ipref = nim1
c	jpref = nj/2
c	kpref = nk/2
c-- fix pressure ref for mcgiil/zink case ------
c	ipref = 31
c	jpref = 2
c	kpref = 20
c------------------------------------------------
        END IF
C--------------------------------------------------------------------
C     RESET TURBULENT INTENSITY IF USING CONSTANT EDDY VISCOSITY MODEL
C--------------------------------------------------------------------
        IF (.NOT.LKETM) TINPR = ZERO
        IF (.NOT.LKETM) TINSC = ZERO 
C-----------------------------------------------------------------------
C     INITIAL OUTPUT
C-----------------------------------------------------------------------
        WRITE (6,*) '    P Reference at I,J,K of:',IPREF,JPREF,KPREF
        IF (LPRIN) THEN
          WRITE (3,22000)
          WRITE (3,23000) FLOWPR
          WRITE (3,24000) FLOWSC
          WRITE (3,26000) VISCOS 
          WRITE (3,28000) PRES
          WRITE (3,30000)
          WRITE (3,32000) DIAPR,DIASC,DIACH,CHLGH,CHHGH,CHWID 
          WRITE (3,34000)
          NMAX = MAX0(NI,NJ,NK)
          DO 180 K = NK1,NK,KSKIP
            WRITE(3,36000) K
            WRITE (3,38000) (I,I = 1,NI)
            DO 170 JJ = NJ1,NJ
              J = NJ+1-JJ
              WRITE (3,40000) J, (PCELL(I,J,K), I = 1,NI)
  170       CONTINUE
  180     CONTINUE
          WRITE (3,42000)
          DO 200 KK = 1,NMAX
            WRITE (3,44000) KK,XX(KK),XU(KK),YY(KK),
     &                      YV(KK),ZZ(KK),ZW(KK)
  200     CONTINUE
          WRITE (3,46000)
        END IF
C-------------------------------------------------------------------------
C     Compressible Flow Error Check
C-------------------------------------------------------------------------
        SOFS = 360.0D0
        COMVEL = PT3*SOFS
        DO 320 K = NK1,NK
          DO 310 J = NJ1,NJ
            DO 300 I = NI1,NI
              IF (UU(I,J,K).GT.COMVEL) LTOFAS = .TRUE.
              IF (VV(I,J,K).GT.COMVEL) LTOFAS = .TRUE.
              IF (WW(I,J,K).GT.COMVEL) LTOFAS = .TRUE.
  300       CONTINUE
  310     CONTINUE
  320   CONTINUE
        IF (LTOFAS) THEN
          IF (LPRIN) WRITE(3,*)' VELOCITIES APPROACHING SPEED OF SOUND '
          WRITE(6,*) ' VELOCITIES APPROACHING SPEED OF SOUND '
        END IF
      ELSE
C-------------------------------------------------------------------------
C     Final Error Checks
C
C     Outlet Recirculation Error Check
C-------------------------------------------------------------------------
        DO 510 K = NK1,NK
          DO 500 J = NJ1,NJ
            IF(UU(NI1,J,K).le.tiny .AND. PCELL(NI1,J,K).EQ.FFIELD)
     &          LOUTRC = .TRUE.
            IF(UU(NI,J,K).le.tiny .AND. PCELL(NI,J,K).EQ.FFIELD)
     &          LOUTRC = .TRUE.
  500     CONTINUE
  510   CONTINUE
        DO 610 K = NK1,NK
          DO 600 I = NI1,NI
            IF(VV(I,NJ1,K).le.tiny .AND. PCELL(I,NJ1,K).EQ.FFIELD) 
     &         LOUTRC = .TRUE.
            IF(VV(I,NJ,K).le.tiny .AND. PCELL(I,NJ,K).EQ.FFIELD) 
     &         LOUTRC = .TRUE.
  600     CONTINUE
  610   CONTINUE  
        DO 710 J = NJ1,NJ
          DO 700 I = NI1,NI
            IF(WW(I,J,NK1).le.tiny .AND. PCELL(I,J,NK1).EQ.FFIELD) 
     &         LOUTRC = .TRUE.
            IF(WW(I,J,NK).le.tiny .AND. PCELL(I,J,NK).EQ.FFIELD) 
     &         LOUTRC = .TRUE.
  700     CONTINUE
  710   CONTINUE    
        IF (LOUTRC) THEN
          IF (LPRIN) WRITE(3,*) ' PROBABLE OUTLET RECIRCULATION '
          WRITE(6,*) ' PROBABLE OUTLET RECIRCULATION '
        END IF
C-------------------------------------------------------------------------
C     Compressible Flow Error Check
C-------------------------------------------------------------------------
        SOFS = 360.0D0
        COMVEL = PT3*SOFS
        DO 820 K = NK1,NK
          DO 810 J = NJ1,NJ
            DO 800 I = NI1,NI
              IF (UU(I,J,K).GT.COMVEL) LTOFAS = .TRUE.
              IF (VV(I,J,K).GT.COMVEL) LTOFAS = .TRUE.
              IF (WW(I,J,K).GT.COMVEL) LTOFAS = .TRUE.
  800       CONTINUE
  810     CONTINUE
  820   CONTINUE
        IF (LTOFAS) THEN
          IF (LPRIN) WRITE(3,*)' VELOCITIES APPROACHING SPEED OF SOUND '
          WRITE(6,*) ' VELOCITIES APPROACHING SPEED OF SOUND '
        END IF
      END IF
C-------------------------------------------------------------------------
C     Formats
C-------------------------------------------------------------------------
20000 FORMAT('          Restart files used:         yes ')
21000 FORMAT('     CFD Calculations turned:         off ')
22000 FORMAT(///////////30X,'***REACTOR PARAMETERS***'///)
23000 FORMAT(/31X,'MASS FLOW RATE OF THE PRIMARY   (kg./s):',
     &       8X,1PE11.3)
24000 FORMAT(/31X,'MASS FLOW RATE OF THE SECONDARY (kg./s):',
     &       8X,1PE11.3)
26000 FORMAT(/31X,'LAMINAR VISCOSITY  (KG/M-S):',
     &      20X,1PE11.3)
28000 FORMAT(/31X,'OUTLET PRESSURE  (N/SQM):',23X,1PE11.3)
30000 FORMAT(///31X,'***REACTOR DIMENSIONS***'//)
32000 FORMAT(/31X,'PRIMARY DIAMETER (M):',28X,1PE12.4/
     & 31X,'SECONDARY DIAMETER (M):',26X,1PE12.4/
     & 31X,'CHAMBER DIAMETER (M):',28X,1PE12.4/
     & 31X,'CHAMBER LENGTH (M):',30X,1PE12.4/
     & 31X,'CHAMBER HEIGHT (M):',30X,1PE12.4/
     & 31X,'REACTOR WIDTH (M):',31X,1PE12.4)
34000 FORMAT(////////35X,'***REACTOR BOUNDARIES AND FLOW DOMAIN***'//)
36000 FORMAT(//30X,' FOR PLANE K = ',I5)
38000 FORMAT(/12X,41(1X,I2))
40000 FORMAT(10X,I2,41(2X,A1))
42000 FORMAT(/////////////33X,'X',14X,'XU',15X,'Y',14X,'YV',15X,'Z',
     &      14X,'ZW',///)
44000 FORMAT(20X,I3,6(4X,1PE12.4))
46000 FORMAT(/////////////)
      RETURN
      END

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:44  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
