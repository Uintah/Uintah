c
c $Id$
c
      subroutine bcscalar(domLo, domHi, idxLo, idxHi, f,
     &     ae,aw,an,as,at,ab,
     &     su,sp,
     &     den, fmixin,
     $     domLoU, domHiU, u,
     &     domLoV, domHiV, v,
     &     domLoW, domHiW, w,
c    &     uinlt,vinlt,winlt,
     $     sew,sns,stb,
     &     pcell, wall, symtry, pfield, ffield, sfield, outletfield)
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
      integer domLo(3), domHi(3), idxLo(3), idxHi(3)
c     integer idxLo_ext(3), idxHi_ext(3)
      double precision f(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision ae(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision aw(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision an(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision as(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision at(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision ab(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision su(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision sp(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision den(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision fmixin
      integer domLoU(3), domHiU(3)
      double precision u(domLoU(1):domHiU(1), domLoU(2):domHiU(2),
     $     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision v(domLoV(1):domHiV(1), domLoV(2):domHiV(2),
     $     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3)
      double precision w(domLoW(1):domHiW(1), domLoW(2):domHiW(2),
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
      integer wall, symtry, pfield, ffield, sfield, outletfield
*
*     declare local variables
*
      double precision areaew, arean, areas, areatb
      integer i, j, k
      logical lwrap, lsymb, ltest, loutsf, liflx, lprdjk, lprdkk
      
c     include 'celltypes.h'

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
     $     f,ae,aw,an,as,at,ab,
     $     pcell, symtry, lprdjk, lprdkk)

      call wallbc(domLo, domHi, idxLo, idxHi, ae,aw,an,as,at,ab,
     $     pcell, wall)

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
c    $                 uinlt(j,k)*
     &                 den(idxHi(1)+1,j,k)*f(idxHi(1)+1,j,k)*
     $                 areaew
               end if
               if (pcell(idxLo(1)-1,j,k).eq.pfield.or.
     $              pcell(idxLo(1)-1,j,k).eq.sfield) then
c                 areaew = ra(j)*sns(j)*stb(k)
                  areaew = sns(j)*stb(k)
                  aw(idxLo(1),j,k) = zero
                  su(idxLo(1),j,k) = su(idxLo(1),j,k) + 
     $                 u(idxLo(1)-1,j,k)*
     &                 den(idxLo(1)-1,j,k)*f(idxLo(1)-1,j,k)*
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
c    $                 vinlt(i,k)*
     &                 den(i,idxHi(2)+1,k)*f(i,idxHi(2)+1,k)*
     $                 arean
               end if
               if (pcell(i,idxLo(2)-1,k).eq.pfield.or.
     $              pcell(i,idxLo(2)-1,k).eq.sfield) then
c                 areas = rv(idxLo(2))*sew(i)*stb(k)
                  areas = sew(i)*stb(k)
                  as(i,idxLo(2),k) = zero
                  su(i,idxLo(2),k) = su(i,idxLo(2),k) + 
     $                 v(i,idxLo(2)-1,k)*
     &                 den(i,idxLo(2)-1,k)*f(i,idxLo(2)-1,k)*
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
c    $                 winlt(i,j)*
     &                 den(i,j,idxHi(3)+1)*f(i,j,idxHi(3)+1)*
     $                 areatb
               end if
               if (pcell(i,j,idxLo(3)-1).eq.pfield.or.
     $              pcell(i,j,idxLo(3)-1).eq.sfield) then
                  areatb = sew(i)*sns(j)
                  ab(i,j,idxLo(3)) = zero
                  su(i,j,idxLo(3)) = su(i,j,idxLo(3)) + 
     $                 w(i,j,idxLo(3)-1)*
     &                 den(i,j,idxLo(3)-1)*f(i,j,idxLo(3)-1)*
     $                 areatb
               end if
 400        continue
 410     continue
      end if
c----------------------------------------------------------------------
c     outlet boundary conditions 
c       east and/or west walls
c----------------------------------------------------------------------
      do 610 k = idxLo(3),idxHi(3)
         do 600 j = idxLo(2),idxHi(2)
          if ((pcell(idxHi(1)+1,j,k).eq.outletfield.and.
     $          pcell(idxHi(1),j,k).eq.ffield)) then
c    &      .or. (pcell(idxHi(1)+1,j,k).eq.decomp.and.
c    $          uinlt(j,k).gt.zero)) then
             if (loutsf) then
                f(idxHi(1)+1,j,k) = f(idxHi(1),j,k)
                ae(idxHi(1),j,k) = two*tiny
             else
                f(idxHi(1)+1,j,k) = fmixin
             end if
          end if
          if ((pcell(idxLo(1)-1,j,k).eq.outletfield.and.
     $         pcell(idxLo(1),j,k).eq.ffield)) then
c    &      .or.(pcell(idxLo(1)-1,j,k).eq.decomp.and.
c    $         u(idxLo(1)-1,j,k).lt.zero)) then
             if (loutsf) then
                f(idxLo(1)-1,j,k) = f(idxLo(1),j,k)
                aw(idxLo(1),j,k) = two*tiny
             else
                f(idxLo(1)-1,j,k) = fmixin
             end if
          end if
  600   continue
  610 continue
c--------------------------------------------------------------------
c     north and/or south walls
c--------------------------------------------------------------------
      do 710 k = idxLo(3),idxHi(3)
         do 700 i = idxLo(1),idxHi(1)
            if ((pcell(i,idxHi(2)+1,k).eq.outletfield.and.
     $           pcell(i,idxHi(2),k).eq.ffield)) then
c    &      .or.(pcell(i,idxHi(2)+1,k).eq.decomp.and.
c    $          vinlt(i,k).gt.zero)) then
             if (loutsf) then
                f(i,idxHi(2)+1,k) = f(i,idxHi(2),k)
                an(i,idxHi(2),k) = two*tiny
             else
                f(i,idxHi(2)+1,k) = fmixin
             end if
          end if
          if ((pcell(i,idxLo(2)-1,k).eq.outletfield.and.
     $         pcell(i,idxLo(2),k).eq.ffield)) then
c    &      .or.(pcell(i,idxLo(2)-1,k).eq.decomp.and.
c    $         v(i,idxLo(2)-1,k).lt.zero)) then
             if (loutsf) then
                f(i,idxLo(2)-1,k) = f(i,idxLo(2),k)
                as(i,idxLo(2),k) = two*tiny
             else
                f(i,idxLo(2)-1,k) = fmixin
             end if
          end if
  700   continue
  710 continue
c--------------------------------------------------------------------
c     top and/or bottom walls
c--------------------------------------------------------------------
      do 810 j = idxLo(2),idxHi(2)
        do 800 i = idxLo(1),idxHi(1)
          if ((pcell(i,j,idxHi(3)+1).eq.outletfield.and.
     $          pcell(i,j,idxHi(3)).eq.ffield)) then
c    &      .or.(pcell(i,j,idxHi(3)+1).eq.decomp.and.
c    $          winlt(i,j).gt.zero)) then
             if (loutsf) then
                f(i,j,idxHi(3)+1) = f(i,j,idxHi(3))
                at(i,j,idxHi(3)) = two*tiny
             else
                f(i,j,idxHi(3)+1) = fmixin
             end if
          end if
          if ((pcell(i,j,idxLo(3)-1).eq.outletfield.and.
     $         pcell(i,j,idxLo(3)).eq.ffield)) then
c    &      .or.(pcell(i,j,idxLo(3)-1).eq.decomp.and.
c    $         w(i,j,idxLo(3)-1).lt.zero)) then
             if (loutsf) then
                f(i,j,idxLo(3)-1) = f(i,j,idxLo(3))
                ab(i,j,idxLo(3)) = two*tiny
             else
                f(i,j,idxLo(3)-1) = fmixin
             end if
          end if
  800   continue
  810 continue
      return
      end

c
c $Log$
c Revision 1.1  2000/07/30 22:21:26  bbanerje
c Added bcscalar.F (originally bcf.f in Kumar's code) needs more work
c in C++ side.
c
c
