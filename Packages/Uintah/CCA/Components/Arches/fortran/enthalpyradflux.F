c
c
      subroutine enthalpyradflux(domLo, domHi, idxLo, idxHi,
     $     qfluxe, qfluxw, qfluxn, qfluxs, qfluxt, qfluxb,
     $     domLoT, domHiT, TG,
     $     domLoA, domHiA, Absorp,
     $     domLoG, domHiG,
     &     dxep, dxpw, dynp, dyps, dztp, dzpb)

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
      integer domLoA(3), domHiA(3)
      double precision Absorp(domLoA(1):domHiA(1), 
     $     domLoA(2):domHiA(2),
     $     domLoA(3):domHiA(3))
      integer domLoG(3), domHiG(3)
      double precision dxep(domLoG(1):domHiG(1))
      double precision dxpw(domLoG(1):domHiG(1))
      double precision dynp(domLoG(2):domHiG(2))
      double precision dyps(domLoG(2):domHiG(2))
      double precision dztp(domLoG(3):domHiG(3))
      double precision dzpb(domLoG(3):domHiG(3))
      
*     local variables

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision radconst, tge, tgw, tgn, tgs, tgt, tgb
      double precision dtdxe, dtdxw, dtdyn, dtdys, dtdzt, dtdzb


      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------

      CALL ARRASS(qfluxe,ZERO,domLo, domHi, idxLo, idxHi)
      CALL ARRASS(qfluxw,ZERO,domLo, domHi, idxLo, idxHi)
      CALL ARRASS(qfluxn,ZERO,domLo, domHi, idxLo, idxHi)
      CALL ARRASS(qfluxs,ZERO,domLo, domHi, idxLo, idxHi)
      CALL ARRASS(qfluxt,ZERO,domLo, domHi, idxLo, idxHi)
      CALL ARRASS(qfluxb,ZERO,domLo, domHi, idxLo, idxHi)

      do 120 k = KST, KEND
         do 110 j = JST, JEND
            do 100 i = IST, IEND
               tge = pt5*(tg(i,j,k)+tg(i+1,j,k))
               tgw = pt5*(tg(i,j,k)+tg(i-1,j,k))
               tgn = pt5*(tg(i,j,k)+tg(i,j+1,k))
               tgs = pt5*(tg(i,j,k)+tg(i,j-1,k))
               tgt = pt5*(tg(i,j,k)+tg(i,j,k+1))
               tgb = pt5*(tg(i,j,k)+tg(i,j,k-1))
               dtdxe = (tg(i+1,j,k)-tg(i,j,k))/dxep(i)
               dtdxw = (tg(i,j,k)-tg(i-1,j,k))/dxpw(i)
               dtdyn = (tg(i,j+1,k)-tg(i,j,k))/dynp(j)
               dtdys = (tg(i,j,k)-tg(i,j-1,k))/dyps(j)
               dtdzt = (tg(i,j,k+1)-tg(i,j,k))/dztp(k)
               dtdzb = (tg(i,j,k)-tg(i,j,k-1))/dzpb(k)
               radconst = -16*SIGMA/(3.0*Absorp(i,j,k)+SMALL)
               qfluxe(i,j,k) = radconst*(tge**3)*dtdxe
               qfluxw(i,j,k) = radconst*(tgw**3)*dtdxw
               qfluxn(i,j,k) = radconst*(tgn**3)*dtdyn
               qfluxs(i,j,k) = radconst*(tgs**3)*dtdys
               qfluxt(i,j,k) = radconst*(tgt**3)*dtdzt
               qfluxb(i,j,k) = radconst*(tgb**3)*dtdzb
 100        continue
 110     continue
 120  continue

      return
      end

