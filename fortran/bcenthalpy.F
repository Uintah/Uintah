c
c
      subroutine bcenthalpy(domLo, domHi,domLong, domHing, 
     $     idxLo, idxHi, H,
     &     ae,aw,an,as,at,ab,
     &     su,sp,
     &     den,
     $     domLoU, domHiU, uu,
     &     domLoV, domHiV, vv,
     &     domLoW, domHiW, ww,
c    &     uinlt,vinlt,winlt,
     $     sew,sns,stb,
     &     pcell, wall, symtry, pfield, presstype, ffield, sfield,
     $     outletfield,
     $     xminus, xplus, yminus, yplus, zminus, zplus)
c    &     r,rv,ra,
c    $     idxLo_ext, idxHi_ext, 
c    $     ltest,lwrap,lsymb,loutsf,
c    $     pcell)
C-----------------------------------------------------------------------
C     THIS SUBROUTINE CALCULATES THE BOUNDARY CONDITONS FOR F
C-----------------------------------------------------------------------
      implicit none

      include 'param4.h'
*
*     declare passed variables
*
      integer domLo(3), domHi(3), domLong(3), domHing(3),
     $     idxLo(3), idxHi(3)
c     integer idxLo_ext(3), idxHi_ext(3)
      double precision H(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision ae(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision aw(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision an(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision as(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision at(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision ab(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision su(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision sp(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision den(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      integer domLoU(3), domHiU(3)
      double precision uu(domLoU(1):domHiU(1), domLoU(2):domHiU(2),
     $     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision vv(domLoV(1):domHiV(1), domLoV(2):domHiV(2),
     $     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3)
      double precision ww(domLoW(1):domHiW(1), domLoW(2):domHiW(2),
     $     domLoW(3):domHiW(3))

c     double precision uinlt(domLo(2):domHi(2),domLo(3):domHi(3))
c     double precision vinlt(domLo(1):domHi(1),domLo(3):domHi(3))
c     double precision winlt(domLo(1):domHi(1),domLo(2):domHi(2))

      double precision sew(domLo(1):domHi(1))
      double precision sns(domLo(2):domHi(2))
      double precision stb(domLo(3):domHi(3))

c     double precision r(domLo(2):domHi(2))
c     double precision rv(domLo(2):domHi(2))
c     double precision ra(domLo(2):domHi(2))
      
      integer pcell(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      integer wall, symtry, pfield, presstype, 
     $        ffield, sfield, outletfield
      integer xminus, xplus, yminus, yplus, zminus, zplus
*
*     declare local variables
*
      double precision areaew, arean, areas, areatb
      integer i, j, k
      logical lwrap, lsymb, ltest, loutsf, liflx, lprdjk, lprdkk
      

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
     $     h,ae,aw,an,as,at,ab,
     $     pcell, symtry, lprdjk, lprdkk)

C  WARNING - should be domLong, domHing - Steve
      call wallbc(domLo, domHi, domLong, domHing, idxLo, idxHi, 
     $     ae,aw,an,as,at,ab,
     $     pcell, wall,
     $     xminus, xplus, yminus, yplus, zminus, zplus)

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
     $                 (uu(idxHi(1)+2,j,k).gt.0.0))) then
                     if ((loutsf).or.
     $                    (pcell(idxHi(1)+1,j,k).eq.presstype)) then
                        H(idxHi(1)+1,j,k) = H(idxHi(1),j,k)
                        ae(idxHi(1),j,k) = two*tiny
                     end if
                  end if
               endif
               if (xminus) then
                  if ((pcell(idxLo(1)-1,j,k).eq.outletfield).or.
     $                 ((pcell(idxLo(1)-1,j,k).eq.presstype).and.
     $                 (uu(idxLo(1)-1,j,k).lt.0.0))) then
                     if ((loutsf).or.
     $                    (pcell(idxLo(1)-1,j,k).eq.presstype)) then
                        H(idxLo(1)-1,j,k) = H(idxLo(1),j,k)
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
     $                 (vv(i,idxHi(2)+2,k).gt.0.0))) then
                     if ((loutsf).or.
     $                    (pcell(i, idxHi(2)+1, k).eq.presstype)) then
                        H(i,idxHi(2)+1,k) = H(i,idxHi(2),k)
                        an(i,idxHi(2),k) = two*tiny
                     end if
                  end if
               endif
               if (yminus) then
                  if ((pcell(i,idxLo(2)-1,k).eq.outletfield).or.
     $                 ((pcell(i,idxLo(2)-1,k).eq.presstype).and.
     $                 (vv(i,idxLo(2)-1,k).lt.0.0))) then
                     if ((loutsf).or.(pcell(i, idxLo(2)-1, k).eq.
     $                    presstype)) then
                        H(i,idxLo(2)-1,k) = H(i,idxLo(2),k)
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
     $                 (ww(i,j,idxHi(3)+2).gt.0.0))) then
                     if ((loutsf).or.(pcell(i, j, idxHi(3)+1).eq.
     $                    presstype)) then
                        H(i,j,idxHi(3)+1) = H(i,j,idxHi(3))
                        at(i,j,idxHi(3)) = two*tiny
                     end if
                  end if
               endif
               if (zminus) then
                  if ((pcell(i,j,idxLo(3)-1).eq.outletfield).or.
     $                 ((pcell(i,j,idxLo(3)-1).eq.presstype).and.
     $                 (ww(i,j,idxLo(3)-1).lt.0.0))) then
                     if ((loutsf).or.(pcell(i, j, idxLo(3)-1).eq.
     $                    presstype)) then
                        H(i,j,idxLo(3)-1) = H(i,j,idxLo(3))
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
