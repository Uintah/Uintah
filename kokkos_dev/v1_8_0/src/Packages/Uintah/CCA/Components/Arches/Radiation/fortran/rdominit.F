      SUBROUTINE rdominit(IST,IEND,JST,JEND,KST,KEND,qince,qincw,qincn,
     &qincs,qinct,qincb,volq)

      IMPLICIT NONE

      integer i,j,k,IST,IEND,JST,JEND,KST,KEND

      double precision qince(IST:IEND,JST:JEND,KST:KEND)
      double precision qincw(IST:IEND,JST:JEND,KST:KEND)
      double precision qincn(IST:IEND,JST:JEND,KST:KEND)
      double precision qincs(IST:IEND,JST:JEND,KST:KEND)
      double precision qinct(IST:IEND,JST:JEND,KST:KEND)
      double precision qincb(IST:IEND,JST:JEND,KST:KEND)
      double precision volq(IST:IEND,JST:JEND,KST:KEND)

	  do 130 k=KST,KEND
	    do 120 j=JST,JEND
	      do 110 i=IST,IEND
	        qince(i,j,k) = 0.0
	        qincw(i,j,k) = 0.0
	        qincn(i,j,k) = 0.0
	        qincs(i,j,k) = 0.0
	        qinct(i,j,k) = 0.0
	        qincb(i,j,k) = 0.0
110	      continue
120	    continue
130	  continue

        do 160 k=KST,KEND
          do 155 j=JST,JEND
            do 150 i=IST,IEND
              volq(i,j,k) = 0.0
150         continue
155       continue
160     continue
        
      return
      end


