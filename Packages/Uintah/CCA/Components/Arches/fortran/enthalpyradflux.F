c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradflux_fort.h>
#include "param4.h"
      
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

      CALL ARRASS(PASS3A(qfluxe)
     & PASS3B(qfluxe),
     & ZERO, idxLo, idxHi)
      CALL ARRASS(PASS3A(qfluxw)
     & PASS3B(qfluxw),
     & ZERO, idxLo, idxHi)
      CALL ARRASS(PASS3A(qfluxn)
     & PASS3B(qfluxn),
     & ZERO, idxLo, idxHi)
      CALL ARRASS(PASS3A(qfluxs)
     & PASS3B(qfluxs),
     & ZERO, idxLo, idxHi)
      CALL ARRASS(PASS3A(qfluxt)
     & PASS3B(qfluxt),
     & ZERO, idxLo, idxHi)
      CALL ARRASS(PASS3A(qfluxb)
     & PASS3B(qfluxb),
     & ZERO, idxLo, idxHi)

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
               radconst = -16.0D0*SIGMA/(3.0D0*Absorp(i,j,k)+SMALL)
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

