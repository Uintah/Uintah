c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/profv_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/param4.h>

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision FLOW, RHO, AVGVELP
      double precision factor

c*********************************************************************
c     Start :
c*********************************************************************
      ist = idxLo(1)
      jst = idxLo(2)
      kst = idxLo(3)
      iend = idxHi(1)
      jend = idxHi(2)
      kend = idxHi(3)

C-------------------------------------------------------------------- 
C     CALCULATION OF U IN PRIMARY STREAMS
C---------------------------------------------------------------------
      FLOW = FLOWPR
      RHO = DENPR
      if (flow.eq.0.0D0) then
      AVGVELP = inletvel
      else
      IF (AREAPR.le.TINY) THEN
        AVGVELP = ZERO
        FLOWPR = ZERO
      ELSE
        AVGVELP = FLOW/(RHO*AREAPR)
      ENDIF
      endif

#include <Packages/Uintah/CCA/Components/Arches/fortran/ramping.h>

c
c     debug to try pure diffusion case: if we don't do this, then
c     since den = 0.0, avgvelp = nan.
c     Commented for general case, 04/23/95, SK
c
c      AVGVELP = 0.1
C--------------------------------------------------------------------
C     EAST AND/OR WEST INLETS
C--------------------------------------------------------------------
      if (xminus.or.xplus) then
         DO 200 K = kst,kend
            DO 190 J = jst,jend
               if (xminus) then
                  IF (PCELL(ist-1,J,K).EQ.PFIELD) THEN
                     UU(ist-1,J,K) = AVGVELP
                     UU(IST,J,K) = factor*AVGVELP
                     VV(ist-1,J,K) = ZERO
                     WW(ist-1,J,K) = ZERO
                  END IF
               endif
               if (xplus) then
                  IF (PCELL(iend+1,J,K).EQ.PFIELD) THEN
                     UU(iend+2,J,K) = -AVGVELP
                     UU(iend+1,J,K) = -factor*AVGVELP
c            UINLT(J,K) = -AVGVELP
c            UU(NI+1,J,K) = -AVGVELP
                     VV(iend+1,J,K) = ZERO
                     WW(iend+1,J,K) = ZERO
                  END IF
               endif
 190        CONTINUE
 200     CONTINUE
      END IF
C--------------------------------------------------------------------
C     NORTH AND/OR SOUTH INLETS
C--------------------------------------------------------------------
      if (yminus.or.yplus) then
         DO 230 K = kst,kend
            DO 220 I = ist,iend
               if (yminus) then
                  IF (PCELL(I,jst-1,K).EQ.PFIELD) THEN
                     VV(I,jst-1,K) = AVGVELP
                     VV(I,JST,K) = factor*AVGVELP
                     WW(I,jst-1,K) = ZERO
                     UU(I,jst-1,K) = ZERO
                  END IF
               endif
               if (yplus) then
                  IF (PCELL(I,jend+1,K).EQ.PFIELD) THEN
                     VV(I,jend+2,K) = -AVGVELP
                     VV(I,jend+1,K) = -factor*AVGVELP
c            VINLT(I,K) = -AVGVELP
c            VV(I,NJ+1,K) = -AVGVELP
                     WW(I,jend+1,K) = ZERO
                     UU(I,jend+1,K) = ZERO
                  END IF
               endif
 220        CONTINUE
 230     CONTINUE
      end if
C--------------------------------------------------------------------
C     TOP AND/OR BOTTOM INLETS
C--------------------------------------------------------------------
      if (zminus.or.zplus) then
         DO 260 J = jst,jend
            DO 250 I = ist,iend
               if (zminus) then
                  IF (PCELL(I,J,kst-1).EQ.PFIELD) THEN
                     WW(I,J,kst-1) = AVGVELP
                     WW(I,J,kst) = factor*AVGVELP
                     VV(I,J,kst-1) = ZERO
                     UU(I,J,kst-1) = ZERO
                  END IF
               endif
               if (zplus) then
                  IF (PCELL(I,J,kend+1).EQ.PFIELD) THEN
                     WW(I,J,kend+2) = -AVGVELP
                     WW(I,J,kend+1) = -factor*AVGVELP
c            WINLT(I,J) = -AVGVELP
c            WW(I,J,NK+1) = -AVGVELP
                     VV(I,J,kend+1) = ZERO
                     UU(I,J,kend+1) = ZERO
                  END IF
               endif
 250        CONTINUE
 260     CONTINUE
      end if
      RETURN
      END

c*********************************************************************
c
c Revision 1.8  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.7  2000/10/05 16:39:49  rawat
c modified bcs for multi-patch
c
c Revision 1.6  2000/09/07 23:07:21  rawat
c fixed some bugs in bc and added pressure solver using petsc
c
c Revision 1.5  2000/08/08 23:34:20  rawat
c fixed some bugs in profv.F and Properties.cc
c
c Revision 1.4  2000/06/30 22:41:19  bbanerje
c Corrected behavior of profv and profscalar
c
c Revision 1.3  2000/06/29 06:22:50  bbanerje
c Updated FCVariable to SFCX, SFCY, SFCZVariables and made corresponding
c changes to profv.  Code is broken until the changes are reflected
c thru all the files.
c
c Revision 1.2  2000/06/20 20:42:38  rawat
c added some more boundary stuff and modified interface to IntVector. Before
c compiling the code you need to update /Core/Geometry/IntVector.h
c
c Revision 1.1  2000/04/12 20:31:49  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
