      SUBROUTINE rqitb (KK,L,IST,IEND,JST,JEND,KST,KEND,sn,zintfc,
     &qinct,qincb,oxi,wt)
C***********************************************************************
C     SUBROUTINE RQITB - calculates flux incident on T-B (z) walls
C     Modified by Gautham 03/02.
C***********************************************************************
      IMPLICIT NONE

      INTEGER I,J,KK,L,sn,IST,IEND,JST,JEND,KST,KEND

      double precision ZINTFC(IST:IEND,JST:JEND,KST:KEND)
      double precision qinct(IST:IEND,JST:JEND,KST:KEND)
      double precision qincb(IST:IEND,JST:JEND,KST:KEND)
      double precision OXI(sn*(sn+2))
      double precision WT(sn*(sn+2))

	IF (OXI(L) .LT. 0) THEN

	      DO 120 J=JST+1,JEND-1
	        DO 110 I=IST+1,IEND-1

	          qincb(I,J,KK-1) = qincb(I,J,KK-1) + WT(L)
     1                            *ABS(OXI(L))*ZINTFC(I,J,KK)

110	        CONTINUE
120	      CONTINUE

	ELSE

	      DO 170 J=JST+1,JEND-1
	        DO 160 I=IST+1,IEND-1

	          qinct(I,J,KK+1) = qinct(I,J,KK+1) + WT(L)
     1                         * ABS(OXI(L)) * ZINTFC(I,J,KK)

160	        CONTINUE
170	      CONTINUE

	END IF

	RETURN
	END
