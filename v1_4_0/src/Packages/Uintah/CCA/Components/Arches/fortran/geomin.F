c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/geomin_fort.h>
#else
      SUBROUTINE geomin(indexLow, indexHigh, PCELL, 
     &     XX, YY, ZZ, 
     &     NSUBD, NDDMN, IDDMAP,
     &     LDDI1, LDDJ1, LDDK1, LDDNI, LDDNJ, LDDNK,
     &     ierr)
C-----------------------------------------------------------------
C
C     THIS SUBROUTINE READS IN THE GEOMETRY INPUT FILE.  VARIABLES
C     READ IN INCLUDE: IDDMAP (DOMAIN DECOMPOSITION MAP - ARRAY 
C                      VARIABLES ARE FOR E,W,N,S,T,B FACES);
C                      X, Y, Z, AND THE PCELL ARRAY
C
C------------------------------------------------------------------

      implicit none

#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer PCELL(indexLow(1):indexHigh(1),indexLow(2):indexHigh(2),
     &     indexLow(3):indexHigh(3))
      double precision XX(indexLow(1):indexHigh(1))
      double precision YY(indexLow(2):indexHigh(2))
      double precision ZZ(indexLow(3):indexHigh(3))
      integer NSUBD, NDDMN
      integer IDDMAP(6)
      logical LDDI1, LDDJ1, LDDK1, LDDNI, LDDNJ, LDDNK
      integer ierr

c*********************************************************************
c     Locals :
c*********************************************************************
      integer i, j, k
      CHARACTER TITLE(3)*80
      LOGICAL LDDCMP, LDDERR
      integer NI, NJ, NK
      integer NX, NY, NZ
      integer IPLANE
      integer iBegGhost, iEndGhost
      integer jBegGhost, jEndGhost
      integer kBegGhost, kEndGhost
 
c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      NX = indexHigh(1)
      NY = indexHigh(2)
      NZ = indexHigh(3)

c*********************************************************************
c     Start :
c*********************************************************************
      OPEN(UNIT=10,FILE='GRID',STATUS='OLD',FORM='FORMATTED')
      READ(10,1000) TITLE(1)
      READ(10,1000) TITLE(2)
C     READ(10,1000) TITLE(3)
      READ(10,*) NSUBD,NDDMN,(IDDMAP(I),I=1,6)
      READ(10,*) NI,NJ,NK
      IF (NI.GT.NX) STOP ' Error: No. of I cells > dimensions'
      IF (NJ.GT.NY) STOP ' Error: No. of J cells > dimensions'
      IF (NK.GT.NZ) STOP ' Error: No. of K cells > dimensions'

c*********************************************************************
c     Reset the indices of interest
c*********************************************************************
      indexHigh(1) = NI
      indexHigh(2) = NJ
      indexHigh(3) = NK
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)

c*********************************************************************
c     Continue reading :
c*********************************************************************
      READ(10,*) (XX(I),I=iBegGhost,iEndGhost)
      READ(10,*) (YY(J),J=jBegGhost,jEndGhost)
      READ(10,*) (ZZ(K),K=kBegGhost,kEndGhost)
      DO 400 I = iBegGhost,iEndGhost
        READ(10,*) IPLANE
        IF (jEndGhost .LE. 50) THEN
          DO 100 K = kBegGhost,kEndGhost
            READ(10,*) (PCELL(I,J,K),J=jBegGhost,jEndGhost)
  100     CONTINUE
        ELSE 
          DO 200 K = kBegGhost,kEndGhost
            READ(10,*) (PCELL(I,J,K),J=jBegGhost,50)
  200     CONTINUE
          DO 300 K = kBegGhost,kEndGhost
            READ(10,*) (PCELL(I,J,K),J=51,jEndGhost)
  300     CONTINUE
        END IF
  400 CONTINUE
      CLOSE (UNIT = 10)
C------------------------------------------------------------------
C
C     ASSIGNMENT OF INPUT VARIABLES TO GAS3D VARIABLES
C
C------------------------------------------------------------------
C------------------------------------------------------------------
C
C	CHECK FOR EXISTANCE OF DOMAIN INTERFACES
C
C------------------------------------------------------------------
	LDDERR = .FALSE.
	LDDCMP = .FALSE.
	LDDI1 = .FALSE.
	LDDNI = .FALSE.
	LDDJ1 = .FALSE.
	LDDNJ = .FALSE.
	LDDK1 = .FALSE.
	LDDNK = .FALSE.
	DO 610 K=kBegGhost,kEndGHost
	  DO 600 J=jBegGhost,jEndGHost
	    IF (PCELL(iBegGhost,J,K).EQ.DECOMP) LDDI1 = .TRUE.
	    IF (PCELL(iEndGhost,J,K).EQ.DECOMP) LDDNI = .TRUE.
600	  CONTINUE
610	CONTINUE
	DO 630 K=kBegGhost,kEndGhost
	  DO 620 I=iBegGhost,iEndGhost
	    IF (PCELL(I,jBegGhost,K).EQ.DECOMP) LDDJ1 = .TRUE.
	    IF (PCELL(I,jEndGhost,K).EQ.DECOMP) LDDNJ = .TRUE.
620	  CONTINUE
630	CONTINUE
	DO 650 J=jBegGhost,jEndGhost
	  DO 640 I=iBegGhost,iEndGhost
	    IF (PCELL(I,J,kBegGhost).EQ.DECOMP) LDDK1 = .TRUE.
	    IF (PCELL(I,J,kEndGhost).EQ.DECOMP) LDDNK = .TRUE.
640	  CONTINUE
650	CONTINUE
	IF (LDDI1.OR.LDDNI.OR.LDDJ1.OR.LDDNJ.OR.LDDK1.OR.LDDNK) 
     1 LDDCMP = .TRUE.
	IF (LDDI1 .AND. IDDMAP(1).EQ.0) LDDERR = .TRUE.
	IF (LDDNI .AND. IDDMAP(2).EQ.0) LDDERR = .TRUE.
	IF (LDDJ1 .AND. IDDMAP(3).EQ.0) LDDERR = .TRUE.
	IF (LDDNJ .AND. IDDMAP(4).EQ.0) LDDERR = .TRUE.
	IF (LDDK1 .AND. IDDMAP(5).EQ.0) LDDERR = .TRUE.
	IF (LDDNK .AND. IDDMAP(6).EQ.0) LDDERR = .TRUE.
	IF (.NOT. LDDERR) THEN
	  WRITE (6,2000)
	ELSE
	  WRITE (6,3000)
	  STOP 'LDDERR in geomin'
	END IF
C
      RETURN
 1000 FORMAT(1A80)
 2000 FORMAT(//16X,'Geometry input file read successfully')
 3000 FORMAT(//16X,'ERROR in domain interface mapping - '
     1       /16X,'cell types and domain assignments are inconsistent')
      END

c*********************************************************************
c
c
c*********************************************************************
