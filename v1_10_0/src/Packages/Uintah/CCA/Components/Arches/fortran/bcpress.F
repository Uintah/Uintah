c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/bcpress_fort.h>
#include "param4.h"
*-----------------------------------------------------------------------
*     This subroutine calculates the boundary
*     conditions for the pressure and pressure correction equations.
*-----------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer IST, JST, KST, IEND, JEND, KEND
      integer i,j,k

      LOGICAL*1 LWRAP,LSYMB, LPRDKK, LPRDAV, LPRDJK

c*********************************************************************
c     Start :
c*********************************************************************
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

c*********************************************************************
c     Set the logicals :
c*********************************************************************
      LSYMB = .false.
      LWRAP = .false.
      LPRDJK = .false.
      LPRDKK = .false.
      LPRDAV = .false.

*----------------------------------------------------------------------
*     intrusion boundary conditions
*----------------------------------------------------------------------

      do 520 k = idxLo(3), idxHi(3)
         do 510 j = idxLo(2), idxHi(2)
            do 500 i = idxLo(1), idxHi(1)
               
               if (pcell(i,j,k).eq.wall) then
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
               endif

 500        continue
 510     continue
 520  continue
*----------------------------------------------------------------------
*     INLET/OUTLET BOUNDARY CONDITIONS 
*       EAST AND/OR WEST WALLS
*----------------------------------------------------------------------
      if (xminus.or.xplus) then
C$DIR SCALAR
         DO 610 K = KST,KEND
C$DIR SCALAR
            DO 600 J = JST,JEND
               IF ((PCELL(IEND+1,J,K).NE.WALL .AND.
     &              PCELL(IEND,J,K).NE.WALL).AND.xplus) then
                  AP(IEND,J,K) = AP(IEND,J,K)+AE(IEND,J,K)
c                  AP(IEND,J,K) = AP(IEND,J,K)-AE(IEND,J,K)
                  AE(IEND,J,K) = ZERO
#ifdef ARCHES_BC_DEBUG
                  i = iend
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
#endif

               END IF
               IF ((PCELL(IST-1,J,K).NE.WALL .AND.
     &              PCELL(IST,J,K).NE.WALL).AND.
     $              xminus) THEN
                  AP(IST,J,K) = AP(IST,J,K)-AW(IST,J,K)
                  AW(IST,J,K) = ZERO
               END IF
               IF ((PCELL(IST-1,J,K).EQ.WALL .AND.
     &              PCELL(IST,J,K).NE.WALL).AND.
     $              xminus) THEN
                  AP(IST,J,K) = AP(IST,J,K)-AW(IST,J,K)
               END IF
 600        CONTINUE
 610     CONTINUE
      endif
*--------------------------------------------------------------------
*     NORTH AND/OR SOUTH WALLS
*--------------------------------------------------------------------
      if (yminus.or.yplus) then
C$DIR SCALAR
         DO 710 K = KST,KEND
C$DIR SCALAR
            DO 700 I = IST,IEND
               IF ((PCELL(I,JEND+1,K).NE.WALL.AND.
     &              PCELL(I,JEND,K).NE.WALL).AND.
     $              yplus) THEN
                  AP(I,JEND,K) = AP(I,JEND,K)+AN(I,JEND,K)
c                  AP(I,JEND,K) = AP(I,JEND,K)-AN(I,JEND,K)
                  AN(I,JEND,K) = ZERO
#ifdef ARCHES_BC_DEBUG
                  j = jend
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
#endif                  
               END IF
               IF ((PCELL(I,JST-1,K).NE.WALL.AND.
     &              PCELL(I,JST,K).NE.WALL).AND.
     $              yminus) THEN
                  AP(I,JST,K) = AP(I,JST,K)+AS(I,JST,K)
c                  AP(I,JST,K) = AP(I,JST,K)-AS(I,JST,K)
                  AS(I,JST,K) = ZERO
#ifdef ARCHES_BC_DEBUG
                  j = jst
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
#endif
               END IF
 700        CONTINUE
 710     CONTINUE
      endif
*--------------------------------------------------------------------
*     TOP AND/OR BOTTOM WALLS
*--------------------------------------------------------------------
      if (zminus.or.zplus) then
C$DIR SCALAR
         DO 810 J = JST,JEND
C$DIR SCALAR
            DO 800 I = IST,IEND
               IF ((PCELL(I,J,KEND+1).NE.WALL .AND. 
     &              PCELL(I,J,KEND).NE.WALL).AND.
     $              zplus)THEN
                  AP(I,J,KEND) = AP(I,J,KEND)+AT(I,J,KEND)
c                  AP(I,J,KEND) = AP(I,J,KEND)-AT(I,J,KEND)
                  AT(I,J,KEND) = ZERO
#ifdef ARCHES_BC_DEBUG
                  k = kend
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
#endif
               END IF
               IF ((PCELL(I,J,KST-1).NE.WALL .AND. 
     &              PCELL(I,J,KST).NE.WALL).AND.
     $              zminus) THEN
                  AP(I,J,KST) = AP(I,J,KST)+AB(I,J,KST)
c                  AP(I,J,KST) = AP(I,J,KST)-AB(I,J,KST)
                  AB(I,J,KST) = ZERO
#ifdef ARCHES_BC_DEBUG
                  k = kst
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
#endif
               END IF
 800        CONTINUE
 810     CONTINUE
      endif
*--------------------------------------------------------------------
*     SYMMETRY AND INTRUSION BOUNDARY CONDITIONS
*--------------------------------------------------------------------
      IF (LSYMB) THEN
         CALL SYMBCS(domLo, domHi, idxLo, idxHi,
     & PASS3A(PP)
     & PASS3B(PP),
     & PASS3A(AE)
     & PASS3B(AE),
     & PASS3A(AW)
     & PASS3B(AW),
     & PASS3A(AN)
     & PASS3B(AN),
     & PASS3A(AS)
     & PASS3B(AS),
     & PASS3A(AT)
     & PASS3B(AT),
     & PASS3A(AB)
     & PASS3B(AB),
     & PASS3A(PCELL)
     & PASS3B(PCELL),
     & PCELL, SYMTRY,LPRDJK, LPRDKK)
      ENDIF

      CALL WALLBC(idxLo, idxHi,
     & PASS3A(AE)
     & PASS3B(AE),
     & PASS3A(AW)
     & PASS3B(AW),
     & PASS3A(AN)
     & PASS3B(AN),
     & PASS3A(AS)
     & PASS3B(AS),
     & PASS3A(AT)
     & PASS3B(AT),
     & PASS3A(AB)
     & PASS3B(AB),
     & PASS3A(PCELL)
     & PASS3B(PCELL),
     & WALL, xminus, xplus, yminus, yplus, zminus, zplus)
      RETURN
      END 

c*********************************************************************
c
c Revision 1.6  2000/10/07 23:53:03  rawat
c modified bc subroutines
c
c Revision 1.5  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.4  2000/09/26 19:59:19  sparker
c Work on MPI petsc
c
c Revision 1.3  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.2  2000/07/17 22:07:02  rawat
c modified momentum source
c
c Revision 1.1  2000/07/13 04:51:34  bbanerje
c Added pressureBC (bcp) .. now called bcpress.F (bcp.F removed)
c
c
c*********************************************************************
