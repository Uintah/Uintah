c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/bcenthalpy_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE BOUNDARY CONDITONS FOR F
C-----------------------------------------------------------------------

*
*     declare local variables
*
      double precision areaew, arean, areas, areatb
      integer i, j, k
      logical*1 lwrap, lsymb, ltest, loutsf, liflx, lprdjk, lprdkk
      

*-----------------------------------------------------------------------

*-----corrections for interior intrusions-------------------------------

      do 305 k=idxLo(3),idxHi(3)
         do 205 j=idxLo(2),idxHi(2)
            do 105 i=idxLo(1),idxHi(1)
               if (pcell(i,j,k).eq.wall) then
                  call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     &                 as(i,j,k),at(i,j,k),ab(i,j,k),su(i,j,k),
     &                 sp(i,j,k),one)
               end if
C               if ((i.eq.0).and.(j.eq.8).and.(k.eq.8)) then
C                  print*, 'Nonlinear source inside bcenthalpy!'
C                  print*, su(i,j,k)
C                  print*,'end bcenthalpy'
C               endif
 105        continue
 205     continue
 305  continue

c--------------------------------------------------------------------
c           inlet conditions at node points are specified in 
c           invar however the flux bc's are specified here!
c           also note that every time a is set to zero c's must
c           be set to zero in calcf.  however, c's in the su terms
c           should not be zeroed at inlets. i.e. if liflx = .true.
c           then lcons must be .true. until i figure a better way!
c
c           symmetry and intrusion boundary conditions
c---------------------------------------------------------------------
      liflx = .false. 
      lsymb = .false.
      lprdjk = .false.
      lprdkk = .false.
      ltest = .false.
      loutsf = .false.

      if (lsymb) call symbcs(domLo, domHi, idxLo, idxHi,
     & PASS3A(h)
     & PASS3B(h),
     & PASS3A(ae)
     & PASS3B(ae),
     & PASS3A(aw)
     & PASS3B(aw),
     & PASS3A(an)
     & PASS3B(an),
     & PASS3A(as)
     & PASS3B(as),
     & PASS3A(at)
     & PASS3B(at),
     & PASS3A(ab)
     & PASS3B(ab),
     & PASS3A(pcell)
     & PASS3B(pcell),
     & symtry, lprdjk, lprdkk)

      call wallbc(idxLo, idxHi,
     & PASS3A(ae)
     & PASS3B(ae),
     & PASS3A(aw)
     & PASS3B(aw),
     & PASS3A(an)
     & PASS3B(an),
     & PASS3A(as)
     & PASS3B(as),
     & PASS3A(at)
     & PASS3B(at),
     & PASS3A(ab)
     & PASS3B(ab),
     & PASS3A(pcell)
     & PASS3B(pcell),
     & wall, xminus, xplus, yminus, yplus, zminus, zplus)

      if (liflx.and..not.ltest) then
c----------------------------------------------------------------------
c     inlet boundary conditions 
c       east and/or west walls
c----------------------------------------------------------------------
         do 210 k = idxLo(3),idxHi(3)
            do 200 j = idxLo(2),idxHi(2)
               if (pcell(idxHi(1)+1,j,k).eq.pfield.or.
     $              pcell(idxHi(1)+1,j,k).eq.sfield) then
c                 areaew = ra(j)*sns(j)*stb(k)
                  areaew = sns(j)*stb(k)
                  ae(idxHi(1),j,k) = zero
                  su(idxHi(1),j,k) = su(idxHi(1),j,k) - 
     $                 uu(idxHi(1)+2,j,k)*
     &                 den(idxHi(1)+1,j,k)*H(idxHi(1)+1,j,k)*
     $                 areaew
               end if
               if (pcell(idxLo(1)-1,j,k).eq.pfield.or.
     $              pcell(idxLo(1)-1,j,k).eq.sfield) then
c                 areaew = ra(j)*sns(j)*stb(k)
                  areaew = sns(j)*stb(k)
                  aw(idxLo(1),j,k) = zero
                  su(idxLo(1),j,k) = su(idxLo(1),j,k) + 
     $                 uu(idxLo(1)-1,j,k)*
     &                 den(idxLo(1)-1,j,k)*H(idxLo(1)-1,j,k)*
     $                 areaew
               end if
 200        continue
 210     continue
c--------------------------------------------------------------------
c     north and/or south walls
c--------------------------------------------------------------------
         do 310 k = idxLo(3),idxHi(3)
            do 300 i = idxLo(1),idxHi(1)
               if (pcell(i,idxHi(2)+1,k).eq.pfield.or.
     $              pcell(i,idxHi(2)+1,k).eq.sfield) then
c                 arean = rv(idxHi(2)+1)*sew(i)*stb(k)
                  arean = sew(i)*stb(k)
                  an(i,idxHi(2),k) = zero
                  su(i,idxHi(2),k) = su(i,idxHi(2),k) - 
     $                 vv(i,idxHi(2)+2,k)*
     &                 den(i,idxHi(2)+1,k)*H(i,idxHi(2)+1,k)*
     $                 arean
               end if
               if (pcell(i,idxLo(2)-1,k).eq.pfield.or.
     $              pcell(i,idxLo(2)-1,k).eq.sfield) then
c                 areas = rv(idxLo(2))*sew(i)*stb(k)
                  areas = sew(i)*stb(k)
                  as(i,idxLo(2),k) = zero
                  su(i,idxLo(2),k) = su(i,idxLo(2),k) + 
     $                 vv(i,idxLo(2)-1,k)*
     &                 den(i,idxLo(2)-1,k)*H(i,idxLo(2)-1,k)*
     $                 areas
               end if
 300        continue
 310     continue
c--------------------------------------------------------------------
c     top and/or bottom walls
c--------------------------------------------------------------------
         do 410 j = idxLo(2),idxHi(2)
            do 400 i = idxLo(1),idxHi(1)
               if (pcell(i,j,idxHi(3)+1).eq.pfield.or.
     &              pcell(i,j,idxHi(3)+1).eq.sfield) then
                  areatb = sew(i)*sns(j)
                  at(i,j,idxHi(3)) = zero
                  su(i,j,idxHi(3)) = su(i,j,idxHi(3)) - 
     $                 ww(i,j,idxHi(3)+2)*
     &                 den(i,j,idxHi(3)+1)*H(i,j,idxHi(3)+1)*
     $                 areatb
               end if
               if (pcell(i,j,idxLo(3)-1).eq.pfield.or.
     $              pcell(i,j,idxLo(3)-1).eq.sfield) then
                  areatb = sew(i)*sns(j)
                  ab(i,j,idxLo(3)) = zero
                  su(i,j,idxLo(3)) = su(i,j,idxLo(3)) + 
     $                 ww(i,j,idxLo(3)-1)*
     &                 den(i,j,idxLo(3)-1)*H(i,j,idxLo(3)-1)*
     $                 areatb
               end if
 400        continue
 410     continue
      end if
c----------------------------------------------------------------------
c     outlet boundary conditions 
c       east and/or west walls
c----------------------------------------------------------------------
      if (xplus.or.xminus) then
         do 610 k = idxLo(3),idxHi(3)
            do 600 j = idxLo(2),idxHi(2)
               if (xplus) then
                  if ((pcell(idxHi(1)+1,j,k).eq.outletfield).or.
     $                 ((pcell(idxHi(1)+1,j,k).eq.presstype).and.
     $                 (uu(idxHi(1)+2,j,k).gt.0.0D0))) then
                     if ((loutsf).or.
     $                    (pcell(idxHi(1)+1,j,k).eq.presstype).or.
     $                    (pcell(idxHi(1)+1,j,k).eq.outletfield)) then
c                        H(idxHi(1)+1,j,k) = H(idxHi(1),j,k)
                        ae(idxHi(1),j,k) = two*tiny
                     end if
                  end if
               endif
               if (xminus) then
                  if ((pcell(idxLo(1)-1,j,k).eq.outletfield).or.
     $                 ((pcell(idxLo(1)-1,j,k).eq.presstype).and.
     $                 (uu(idxLo(1)-1,j,k).lt.0.0D0))) then
                     if ((loutsf).or.
     $                    (pcell(idxLo(1)-1,j,k).eq.presstype)) then
c                        H(idxLo(1)-1,j,k) = H(idxLo(1),j,k)
                        aw(idxLo(1),j,k) = two*tiny
                     end if
                  end if
               endif
 600        continue
 610     continue
      endif
c--------------------------------------------------------------------
c     north and/or south walls
c--------------------------------------------------------------------
      if (yminus.or.yplus) then
         do 710 k = idxLo(3),idxHi(3)
            do 700 i = idxLo(1),idxHi(1)
               if (yplus) then
                  if ((pcell(i,idxHi(2)+1,k).eq.outletfield).or.
     $                 ((pcell(i,idxHi(2)+1,k).eq.presstype).and.
     $                 (vv(i,idxHi(2)+2,k).gt.0.0D0))) then
                     if ((loutsf).or.
     $                    (pcell(i, idxHi(2)+1, k).eq.presstype)) then
c                        H(i,idxHi(2)+1,k) = H(i,idxHi(2),k)
                        an(i,idxHi(2),k) = two*tiny
                     end if
                  end if
               endif
               if (yminus) then
                  if ((pcell(i,idxLo(2)-1,k).eq.outletfield).or.
     $                 ((pcell(i,idxLo(2)-1,k).eq.presstype).and.
     $                 (vv(i,idxLo(2)-1,k).lt.0.0D0))) then
                     if ((loutsf).or.(pcell(i, idxLo(2)-1, k).eq.
     $                    presstype)) then
c                        H(i,idxLo(2)-1,k) = H(i,idxLo(2),k)
                        as(i,idxLo(2),k) = two*tiny
                     end if
                  end if
               endif
 700        continue
 710     continue
      endif
c--------------------------------------------------------------------
c     top and/or bottom walls
c--------------------------------------------------------------------
      if (zminus.or.zplus) then
         do 810 j = idxLo(2),idxHi(2)
            do 800 i = idxLo(1),idxHi(1)
               if (zplus) then
                  if ((pcell(i,j,idxHi(3)+1).eq.outletfield).or.
     $                 ((pcell(i,j,idxHi(3)+1).eq.presstype).and.
     $                 (ww(i,j,idxHi(3)+2).gt.0.0D0))) then
                     if ((loutsf).or.(pcell(i, j, idxHi(3)+1).eq.
     $                    presstype)) then
c                        H(i,j,idxHi(3)+1) = H(i,j,idxHi(3))
                        at(i,j,idxHi(3)) = two*tiny
                     end if
                  end if
               endif
               if (zminus) then
                  if ((pcell(i,j,idxLo(3)-1).eq.outletfield).or.
     $                 ((pcell(i,j,idxLo(3)-1).eq.presstype).and.
     $                 (ww(i,j,idxLo(3)-1).lt.0.0D0))) then
                     if ((loutsf).or.(pcell(i, j, idxLo(3)-1).eq.
     $                    presstype)) then
c                        H(i,j,idxLo(3)-1) = H(i,j,idxLo(3))
                        ab(i,j,idxLo(3)) = two*tiny
                     end if
                  end if
               endif
 800        continue
 810     continue
      endif
      return
      end

c
c Revision 1.7  2000/10/10 19:31:03  rawat
c added scalarsolver
c
c Revision 1.6  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.5  2000/09/26 19:59:19  sparker
c Work on MPI petsc
c
c Revision 1.4  2000/09/07 23:07:21  rawat
c fixed some bugs in bc and added pressure solver using petsc
c
c Revision 1.3  2000/08/19 16:36:40  rawat
c fixed some bugs in scalarcoef calculations
c
c Revision 1.2  2000/08/09 20:19:27  rawat
c modified scalcoef.F
c
c Revision 1.1  2000/07/30 22:21:26  bbanerje
c Added bcscalar.F (originally bcf.f in Kumar's code) needs more work
c in C++ side.
c
c
