c
c

#include <Packages/Uintah/CCA/Components/Arches/fortran/pressrc_fort.h>
#include "param4.h"
      
*     local variables

      double precision arean, areas, areaew, areatb, vol
      double precision dene, denw, denn, dens, dent, denb
      double precision ue, uw, vn, vs, wt, wb
      double precision ce, cw, cn, cs, ct, cb
      double precision smp, apo
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      logical*1 ltest, ltim

c     include 'celltypes.h'
      ltim = .true.
      ltest = .false.
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------

      do 120 k = KST, KEND
         do 110 j = JST, JEND
            do 100 i = IST, IEND

               arean = sew(i)*stb(k)
               areas = sew(i)*stb(k)
               areaew = sns(j)*stb(k)
               areatb = sns(j)*sew(i)
               vol = sns(j)*sew(i)*stb(k)
c-----------------------------------------------------------------------
c     calculate coefficients
c-----------------------------------------------------------------------

               dene = pt5*(den(i,j,k)+den(i+1,j,k))
               denw = pt5*(den(i,j,k)+den(i-1,j,k))
               denn = pt5*(den(i,j,k)+den(i,j+1,k))
               dens = pt5*(den(i,j,k)+den(i,j-1,k))
               dent = pt5*(den(i,j,k)+den(i,j,k+1))
               denb = pt5*(den(i,j,k)+den(i,j,k-1))

c-----------------------------------------------------------------------
c     calculate pseudovelocities for boundaries 
c     (setting uhat = u on boundaries makes it easy to set bcs. for
c     the pressure)
c-----------------------------------------------------------------------

               ue = uu(i+1,j,k)*dene
               uw = uu(i,j,k)*denw
               vn = vv(i,j+1,k)*denn
               vs = vv(i,j,k)*dens
               wt = ww(i,j,k+1)*dent
               wb = ww(i,j,k)*denb
               
c-----------------------------------------------------------------------
c     calculate pseudovelocities when not on boundaries
c for multi-patch include ghost cell of size 1 for pcell and uu
c-----------------------------------------------------------------------

               if (pcell(i-1,j,k).eq.ffield)
     &              uw = (aue(i,j,k)*uu(i+1,j,k)+
     $                    auw(i,j,k)*uu(i-1,j,k)+
     &                    aun(i,j,k)*uu(i,j+1,k)+
     $                    aus(i,j,k)*uu(i,j-1,k)+
     &                    aut(i,j,k)*uu(i,j,k+1)+
     $                    aub(i,j,k)*uu(i,j,k-1)+
     &                    suu(i,j,k)-aup(i,j,k)*uu(i,j,k))*deltat/
     $                    (sewu(i)*sns(j)*stb(k))
               if (pcell(i+1,j,k).eq.ffield)
     &              ue = (aue(i+1,j,k)*uu(i+2,j,k)+
     &                    auw(i+1,j,k)*uu(i,j,k)+
     $                    aun(i+1,j,k)*uu(i+1,j+1,k)+
     &                    aus(i+1,j,k)*uu(i+1,j-1,k)+
     $                    aut(i+1,j,k)*uu(i+1,j,k+1)+
     &                    aub(i+1,j,k)*uu(i+1,j,k-1)+
     $                    suu(i+1,j,k)-aup(i+1,j,k)*uu(i+1,j,k))*
     $                    deltat/(sewu(i+1)*sns(j)*stb(k))
               if (pcell(i,j-1,k).eq.ffield)
     &              vs = (ave(i,j,k)*vv(i+1,j,k)+
     $                    avw(i,j,k)*vv(i-1,j,k)+
     &                    avn(i,j,k)*vv(i,j+1,k)+
     $                    avs(i,j,k)*vv(i,j-1,k)+
     &                    avt(i,j,k)*vv(i,j,k+1)+
     $                    avb(i,j,k)*vv(i,j,k-1)+
     &                    svu(i,j,k)-avp(i,j,k)*vv(i,j,k))*
     $                    deltat/(sew(i)*stb(k)*snsv(j))
               if (pcell(i,j+1,k).eq.ffield)
     &              vn = (ave(i,j+1,k)*vv(i+1,j+1,k)+
     &                    avw(i,j+1,k)*vv(i-1,j+1,k)+
     $                    avn(i,j+1,k)*vv(i,j+2,k)+
     &                    avs(i,j+1,k)*vv(i,j,k)+
     $                    avt(i,j+1,k)*vv(i,j+1,k+1)+
     &                    avb(i,j+1,k)*vv(i,j+1,k-1)+
     $                    svu(i,j+1,k)-avp(i,j+1,k)*vv(i,j+1,k))*
     $                    deltat/(sew(i)*stb(k)*snsv(j+1))
               if (pcell(i,j,k-1).eq.ffield)
     &              wb = (awe(i,j,k)*ww(i+1,j,k)+
     $                    aww(i,j,k)*ww(i-1,j,k)+
     &                    awn(i,j,k)*ww(i,j+1,k)+
     $                    aws(i,j,k)*ww(i,j-1,k)+
     &                    awt(i,j,k)*ww(i,j,k+1)+
     $                    awb(i,j,k)*ww(i,j,k-1)+
     &                    swu(i,j,k)-awp(i,j,k)*ww(i,j,k))*
     $                    deltat/(sew(i)*stbw(k)*sns(j))
               if (pcell(i,j,k+1).eq.ffield)
     &              wt = (awe(i,j,k+1)*ww(i+1,j,k+1)+
     &                    aww(i,j,k+1)*ww(i-1,j,k+1)+
     $                    awn(i,j,k+1)*ww(i,j+1,k+1)+
     &                    aws(i,j,k+1)*ww(i,j-1,k+1)+
     $                    awt(i,j,k+1)*ww(i,j,k+2)+
     &                    awb(i,j,k+1)*ww(i,j,k)+
     $                    swu(i,j,k+1)-awp(i,j,k+1)*ww(i,j,k+1))*   
     $                    deltat/(sew(i)*stbw(k+1)*sns(j))

c-----calculate source terms--------------------------------------------

               ce = areaew
               cw = areaew
               cn = arean
               cs = areas
               ct = areatb
               cb = areatb
               apo = 0.0
               if (ltim) then
                  if (pcell(i,j,k).eq.ffield) 
     $                 apo = (old_den(i,j,k) - den(i,j,k))*vol/deltat
               endif
               smp = (cn*vn-cs*vs+ce*ue-cw*uw+ct*wt-cb*wb - apo)/deltat
c    $              -sign*spm(i,j,k)

#ifdef ARCHES_PRES_DEBUG
               if ((i.eq.0).and.(j.eq.0).and.(k.eq.0)) then
                  print*,'TESTING PRESSURE SOURCE'
                  print*,'den(i,j,k)=',den(i,j,k)
                  print*,'den(i-1,j,k)=',den(i-1,j,k)
                  print*,'smp = ', smp
                  print*,'ue = ',ue
                  print*,'uw = ',uw
                  print*,'vn = ',vn
                  print*,'vs = ',vs
                  print*,'wt = ',wt
                  print*,'wb = ',wb
                  print*,'ce = ',ce
                  print*,'cw = ',cw
                  print*,'cn = ',cn
                  print*,'cs = ',cs
                  print*,'ct = ',ct
                  print*,'cb = ',cb
                  print*,'apo = ',apo
                  print*,'aupp =',aup(i,j,k)
                  print*,'aupe =',aup(i+1,j,k)
                  print*,'avpp =',avp(i,j,k)
                  print*,'avpn =',avp(i,j+1,k)
                  print*,'awpp =',awp(i,j,k)
                  print*,'awpt =',awp(i,j,k+1)
               endif
#endif

               if (ltest) then
                  sp(i,j,k) = zero
c                 su(i,j,k) = -smp + cpl(i,j,k)*vol
               else
                  sp(i,j,k) = zero
                  su(i,j,k) = -smp
c                 if (smp.ne.zero)
c    &                 convs = convs + abs(smp)/max(abs(cn*vn),
c    $                 abs(cs*vs),abs(ce*ue),abs(cw*uw),
c    &                 abs(ct*wt),abs(cb*wb),abs(spm(i,j,k)))
               end if

 100           continue
 110        continue
 120     continue

c        convs = convs/((KEND-KST+1)*(JEND-JST+1)*(IEND-IST+1))

         return
         end
c
c Revision 1.9  2000/10/12 20:07:41  sparker
c Commented out debug print statements
c Fixed bug in profscalar
c
c Revision 1.8  2000/10/07 23:53:03  rawat
c modified bc subroutines
c
c Revision 1.7  2000/10/04 16:46:29  rawat
c Parallel solver for pressure is working
c
c Revision 1.6  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.5  2000/08/23 06:20:53  bbanerje
c 1) Results now correct for pressure solve.
c 2) Modified BCU, BCV, BCW to add stuff for pressure BC.
c 3) Removed some bugs in BCU, V, W.
c 4) Coefficients for MOM Solve not computed correctly yet.
c
c Revision 1.4  2000/08/20 22:52:38  bbanerje
c Fixed PressureSource bug .. domHi, domLo were not being assigned
c correctly in the wrapper.
c
c Revision 1.3  2000/08/17 20:32:02  rawat
c Fixed some bugs
c
c Revision 1.2  2000/07/17 22:07:03  rawat
c modified momentum source
c
c Revision 1.1  2000/07/12 23:23:25  bbanerje
c Added pressure source .. modified Kumar's version a bit.
c
c
