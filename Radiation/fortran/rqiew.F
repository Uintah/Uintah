      SUBROUTINE rqiew(II,JJ,KK,L,IST,IEND,JST,JEND,KST,KEND,sn,xintfc,
     &qince,qincw,omu,wt)
C***********************************************************************
C	SUBROUTINE RQIEW - calculates flux incident on E-W (x) walls
C       Modified by Gautham 03/02.
C***********************************************************************
      IMPLICIT NONE  

      INTEGER II,JJ,KK,L,sn,IST,IEND,JST,JEND,KST,KEND    
      
      double precision XINTFC(IST:IEND,JST:JEND,KST:KEND)
      double precision qince(IST:IEND,JST:JEND,KST:KEND)
      double precision qincw(IST:IEND,JST:JEND,KST:KEND)
      double precision OMU(sn*(sn+2))
      double precision WT(sn*(sn+2))

C	Heat flux incident on E-W (x) faces
C	LR, MR, NR map boundary intensities for periodic bc's

	         IF (OMU(L) .LT. 0) THEN

	              qince(II-1,JJ,KK) = qince(II-1,JJ,KK) + WT(L) 
     1                          * ABS(OMU(L))*XINTFC(II,JJ,KK)

	         ELSE

	              qincw(II+1,JJ,KK) = qincw(II+1,JJ,KK) + WT(L) 
     1                          * ABS(OMU(L))*XINTFC(II,JJ,KK)
	          
                 END IF

                 RETURN

                 END


