c
c
      subroutine pressrcpred(domLo, domHi, domLong, domHing, 
     $     idxLo, idxHi, 
     $     su,  
     &     domLoDen, domHiDen, den,
     &     domLoDenO, domHiDenO, den_old,
     &     domLoU, domHiU, uhat,
     &     domLoV, domHiV, vhat,
     &     domLoW, domHiW, what,
     &     delta_t,
     &     sew, sns, stb)


      implicit none
      INCLUDE 'param4.h'

*     passed variables

      integer domLo(3), domHi(3)
      integer domLong(3), domHing(3)
      integer idxLo(3), idxHi(3)
      double precision su(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      integer domLoDen(3), domHiDen(3)
      double precision den(domLoDen(1):domHiDen(1), 
     $     domLoDen(2):domHiDen(2),
     $     domLoDen(3):domHiDen(3))
      integer domLoDenO(3), domHiDenO(3)
      double precision den_old(domLoDenO(1):domHiDenO(1), 
     $     domLoDenO(2):domHiDenO(2),
     $     domLoDenO(3):domHiDenO(3))
      integer domLoU(3), domHiU(3)
      double precision uhat(domLoU(1):domHiU(1), domLoU(2):domHiU(2),
     $     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision vhat(domLoV(1):domHiV(1), domLoV(2):domHiV(2),
     $     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3)
      double precision what(domLoW(1):domHiW(1), domLoW(2):domHiW(2),
     $     domLoW(3):domHiW(3))
      double precision delta_t
      double precision sew(domLo(1):domHi(1))
      double precision sns(domLo(2):domHi(2))
      double precision stb(domLo(3):domHi(3))

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision dene, denw, denn, dens, dent, denb
      double precision ce, cw, cn, cs, ct, cb
      double precision arean, areas, areaew, areatb, vol
      double precision apo, smp

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
               dene = pt5*(den_old(i+1,j,k)+den_old(i,j,k))
               denw = pt5*(den_old(i-1,j,k)+den_old(i,j,k))
               denn = pt5*(den_old(i,j+1,k)+den_old(i,j,k))
               dens = pt5*(den_old(i,j-1,k)+den_old(i,j,k))
               dent = pt5*(den_old(i,j,k+1)+den_old(i,j,k))
               denb = pt5*(den_old(i,j,k-1)+den_old(i,j,k))
               ce = areaew*dene
               cw = areaew*denw
               cn = arean*denn
               cs = areas*dens
               ct = areatb*dent
               cb = areatb*denb
               apo = (den_old(i,j,k) - den(i,j,k))*vol/delta_t
               smp = (cn*vhat(i,j+1,k)-cs*vhat(i,j,k)+
     $                ce*uhat(i+1,j,k)-cw*uhat(i,j,k)+
     $                ct*what(i,j,k+1)-cb*what(i,j,k) - apo)/delta_t
c    $              -sign*spm(i,j,k)
               su(i,j,k)=-smp
 100           continue
 110        continue
 120     continue

         return
         end

