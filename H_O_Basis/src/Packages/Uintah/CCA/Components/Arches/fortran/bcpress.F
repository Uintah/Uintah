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

      if (curr_bc_type.eq.wall) then
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
      end if
*----------------------------------------------------------------------
*     INLET/OUTLET BOUNDARY CONDITIONS 
*       EAST AND/OR WEST WALLS
*----------------------------------------------------------------------
      if (xminus.or.xplus) then
         DO 610 K = KST,KEND
            DO 600 J = JST,JEND
              IF ((PCELL(IEND+1,J,K).eq.curr_bc_type).AND.xplus) THEN
                 AP(IEND,J,K) = AP(IEND,J,K)+curr_bc_factor*AE(IEND,J,K)
                 AE(IEND,J,K) = ZERO
#ifdef ARCHES_BC_DEBUG
                  i = iend
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
#endif

               END IF
               IF ((PCELL(IST-1,J,K).eq.curr_bc_type).AND.xminus) THEN
                  AP(IST,J,K) = AP(IST,J,K)+curr_bc_factor*AW(IST,J,K)
                  AW(IST,J,K) = ZERO
#ifdef ARCHES_BC_DEBUG
                  i = ist
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
#endif
               END IF
 600        CONTINUE
 610     CONTINUE
      endif
*--------------------------------------------------------------------
*     NORTH AND/OR SOUTH WALLS
*--------------------------------------------------------------------
      if (yminus.or.yplus) then
         DO 710 K = KST,KEND
            DO 700 I = IST,IEND
              IF ((PCELL(I,JEND+1,K).eq.curr_bc_type).AND.yplus) THEN
                 AP(I,JEND,K) = AP(I,JEND,K)+curr_bc_factor*AN(I,JEND,K)
                 AN(I,JEND,K) = ZERO
#ifdef ARCHES_BC_DEBUG
                  j = jend
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
#endif                  
               END IF
               IF ((PCELL(I,JST-1,K).eq.curr_bc_type).AND.yminus) THEN
                  AP(I,JST,K) = AP(I,JST,K)+curr_bc_factor*AS(I,JST,K)
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
         DO 810 J = JST,JEND
            DO 800 I = IST,IEND
              IF ((PCELL(I,J,KEND+1).eq.curr_bc_type).AND.zplus)THEN
                 AP(I,J,KEND) = AP(I,J,KEND)+curr_bc_factor*AT(I,J,KEND)
                 AT(I,J,KEND) = ZERO
#ifdef ARCHES_BC_DEBUG
                  k = kend
                  call fixval(
     $                 zero, ae(i,j,k), aw(i,j,k), an(i,j,k), as(i,j,k),
     $                 at(i,j,k), ab(i,j,k),
     $                 su(i,j,k), sp(i,j,k), one)
#endif
               END IF
               IF ((PCELL(I,J,KST-1).eq.curr_bc_type).AND.zminus) THEN
                  AP(I,J,KST) = AP(I,J,KST)+curr_bc_factor*AB(I,J,KST)
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
c commented symmtry out for the time being
c      IF (LSYMB) THEN
c         CALL SYMBCS(domLo, domHi, idxLo, idxHi,
c     & PASS3A(PP)
c     & PASS3B(PP),
c     & PASS3A(AE)
c     & PASS3B(AE),
c     & PASS3A(AW)
c     & PASS3B(AW),
c     & PASS3A(AN)
c     & PASS3B(AN),
c     & PASS3A(AS)
c     & PASS3B(AS),
c     & PASS3A(AT)
c     & PASS3B(AT),
c     & PASS3A(AB)
c     & PASS3B(AB),
c     & PASS3A(PCELL)
c     & PASS3B(PCELL),
c     & PCELL, SYMTRY,LPRDJK, LPRDKK)
c      ENDIF

c added what wallbc does to the loop explicitly above
c      CALL WALLBC(idxLo, idxHi,
c     & PASS3A(AE)
c     & PASS3B(AE),
c     & PASS3A(AW)
c     & PASS3B(AW),
c     & PASS3A(AN)
c     & PASS3B(AN),
c     & PASS3A(AS)
c     & PASS3B(AS),
c     & PASS3A(AT)
c     & PASS3B(AT),
c     & PASS3A(AB)
c     & PASS3B(AB),
c     & PASS3A(PCELL)
c     & PASS3B(PCELL),
c     & WALL, xminus, xplus, yminus, yplus, zminus, zplus)
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
