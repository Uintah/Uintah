c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/areain_fort.h>
#include "param4.h"
C---------------------------------------------------------------------
C     THIS SUBROUTINE WILL CALCULATE INLET
C     AREAS FOR THE PRIMARY (AREAPR) AND SECONDARY (AREASC)
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision AREAT

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = domainLow(1)
      jBegGhost = domainLow(2)
      kBegGhost = domainLow(3)
      iEndGhost = domainHigh(1)
      jEndGhost = domainHigh(2)
      kEndGhost = domainHigh(3)
      IST = indexLow(1)
      JST = indexLow(2)
      KST = indexLow(3)
      IEND = indexHigh(1)
      JEND = indexHigh(2)
      KEND = indexHigh(3)

C-------------------------------------------------------------------- 
C     CALCULATION OF THE AREA OF TOTAL PRIMARY STREAMS
C--------------------------------------------------------------------
      AREAT = ZERO
      if (xminus.or.xplus) then
         DO 110 K = KST,KEND
            DO 100 J = JST,JEND
               IF ((PCELL(iBegGhost,J,K).EQ.PFIELD).and.xminus)THEN
c           write(*,'("i=",i2," j=",i2," k=",i2," sns=",f5.2," stb=",f5.2)')
c    &        iBegGhost,j,k,sns(j),stb(k) 
                  if (PCELL(iBegGhost+1,J,K).eq.ffield) then
                    AREAT = AREAT + SNS(J)*STB(K)
                  end if
               END IF
               IF ((PCELL(iEndGhost,J,K).EQ.PFIELD).and.xplus) THEN
                  if (PCELL(iEndGhost-1,J,K).eq.ffield) then
                    AREAT = AREAT + SNS(J)*STB(K)
                  end if
               END IF
 100        CONTINUE
 110     CONTINUE
      endif
      if (yminus.or.yplus) then
         DO 140 K = KST,KEND
            DO 130 I = IST,IEND
               IF ((PCELL(I,jBegGhost,K).EQ.PFIELD).and.yminus)THEN
                  if (PCELL(I,jBegGhost+1,K).eq.ffield) then
                    AREAT = AREAT + SEW(I)*STB(K)
c               IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
                  end if
             END IF
             IF ((PCELL(I,jEndGhost,K).EQ.PFIELD).and.yplus) THEN
                  if (PCELL(I,jEndGhost-1,K).eq.ffield) then
                    AREAT = AREAT + SEW(I)*STB(K)
                  end if
             END IF
 130      CONTINUE
 140   CONTINUE
      endif

      if (zminus.or.zplus) then
         DO 170 J = JST,JEND
            DO 160 I = IST,IEND
               IF ((PCELL(I,J,kBegGhost).EQ.PFIELD).and.zminus)THEN
                  if (PCELL(I,J,kBegGhost+1).eq.ffield) then
                    AREAT = AREAT + SEW(I)*SNS(J)
c     IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
                  end if
               END IF
               IF ((PCELL(I,J,kEndGhost).EQ.PFIELD).and.zplus) THEN
                  if (PCELL(I,J,kEndGhost-1).eq.ffield) then
                    AREAT = AREAT + SEW(I)*SNS(J)
                  end if
c     IF(LCORD.AND.LPRIN)WRITE(3,*)' ERROR ON INLET CONDITION'
               END IF
 160        CONTINUE
 170     CONTINUE
      end if
       IF (AREAT.EQ.ZERO) THEN
c         IF(LPRIN) WRITE(3,*)
c    &              ' NO PRIMARY INLET AREA IN INPUT GEOMETRY '
          WRITE(*,*) ' NO PRIMARY INLET AREA IN INPUT GEOMETRY '
       ELSE
          AREAPR = AREAT
       END IF

       RETURN
       END

c*********************************************************************
c
c Revision 1.6  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.5  2000/06/30 06:29:43  bbanerje
c Got Inlet Area to be calculated correctly .. but now two CellInformation
c variables are being created (Rawat ... check that).
c
c Revision 1.4  2000/06/29 06:22:50  bbanerje
c Updated FCVariable to SFCX, SFCY, SFCZVariables and made corresponding
c changes to profv.  Code is broken until the changes are reflected
c thru all the files.
c
c Revision 1.3  2000/06/15 23:47:59  rawat
c modified Archesfort to fix function call
c
c Revision 1.2  2000/06/15 22:13:24  rawat
c modified boundary stuff
c
c Revision 1.1  2000/04/12 20:31:41  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
