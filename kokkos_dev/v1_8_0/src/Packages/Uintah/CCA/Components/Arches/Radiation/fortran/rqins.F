      SUBROUTINE rqins(JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,YINTFC,
     &qincn,qincs,OETA,WT)
C***********************************************************************
C	SUBROUTINE RQINS - calculates flux incident on N-S (y) walls
C       Modified by Gautham 03/02.
C***********************************************************************
      IMPLICIT NONE

      INTEGER I,JJ,KK,L,sn,IST,IEND,JST,JEND,KST,KEND

      double precision YINTFC(IST:IEND,JST:JEND,KST:KEND)
      double precision qincn(IST:IEND,JST:JEND,KST:KEND)
      double precision qincs(IST:IEND,JST:JEND,KST:KEND)
      double precision OETA(sn*(sn+2))
      double precision WT(sn*(sn+2))

C	Heat flux incident on N-S (y) faces - 4 is North cell face;
C	LR, MR, NR map boundary intensities for periodic bc's

	IF (OETA(L) .LT. 0) THEN

	    DO 110 I=IST+1,IEND-1

	      qincn(I,JJ-1,KK) = qincn(I,JJ-1,KK) + WT(L)
     1                    *ABS(OETA(L))*YINTFC(I,JJ,KK)

110	    CONTINUE

	ELSE

	    DO 150 I=IST+1,IEND-1

	      qincs(I,JJ+1,KK) = qincs(I,JJ+1,KK) + WT(L)
     1                * ABS(OETA(L)) * YINTFC(I,JJ,KK)

150	    CONTINUE

	END IF

	RETURN
	END






