c
c
      subroutine enthalpyradwallbc(domLo, domHi, idxLo, idxHi,
     $     qfluxe, qfluxw, qfluxn, qfluxs, qfluxt, qfluxb,
     $     domLoT, domHiT, TG,
     $     domLoP, domHiP, PCELL,
     $     WALLID,
     $     xminus, xplus, yminus, yplus, zminus, zplus)

      implicit none
      INCLUDE 'param4.h'

*     passed variables

      integer domLo(3), domHi(3)
      integer idxLo(3), idxHi(3)
      double precision qfluxe(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision qfluxw(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision qfluxn(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision qfluxs(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision qfluxt(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision qfluxb(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      integer domLoT(3), domHiT(3)
      double precision TG(domLoT(1):domHiT(1), 
     $     domLoT(2):domHiT(2),
     $     domLoT(3):domHiT(3))
      integer domLoP(3), domHiP(3)
      double precision PCELL(domLoP(1):domHiP(1), 
     $     domLoP(2):domHiP(2),
     $     domLoP(3):domHiP(3))
      integer WALLID
      integer xminus, xplus, yminus, yplus, zminus, zplus

      
*     local variables

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision radwallconst
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------
      if (xminus) then
         do 120 k = KST, KEND
            do 110 j = JST, JEND
               i = ist -1
               if (pcell(i,j,k).eq.wallid) then
c     taken from pg 10-16 of fluent manual, assuming nw < .01
                  radwallconst = -SIGMA*TWO
                  qfluxw(i+1,j,k) = radwallconst*
     $                            (tg(i,j,k)**4-tg(i+1,j,k)**4)
               endif
 110        continue
 120     continue
      endif
    
      return
      end

