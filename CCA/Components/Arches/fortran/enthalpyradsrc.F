c
c
      subroutine enthalpyradsrc(domLoS, domHiS, idxLo, idxHi,
     $     SU,
     $     domLo, domHi,
     $     qfluxe, qfluxw, qfluxn, qfluxs, qfluxt, qfluxb,
     $     domLoG, domHiG,
     $     sew, sns, stb)

      implicit none
      INCLUDE 'param4.h'

*     passed variables

      integer domLoS(3), domHiS(3)
      integer idxLo(3), idxHi(3)
      double precision su(domLoS(1):domHiS(1), domLoS(2):domHiS(2),
     $     domLoS(3):domHiS(3))
      integer domLo(3), domHi(3)
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
      integer domLoG(3), domHiG(3)
      double precision sew(domLoG(1):domHiG(1))
      double precision sns(domLoG(2):domHiG(2))
      double precision stb(domLoG(3):domHiG(3))

      
*     local variables

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision arean, areaew, areatb


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
               areaew = sns(j)*stb(k)
               areatb = sns(j)*sew(i)
               su(i,j,k) = su(i,j,k) + 
     $                    areaew*(qfluxw(i,j,k)-qfluxe(i,j,k)) +
     $                    arean*(qfluxs(i,j,k)-qfluxn(i,j,k)) +
     $                    areatb*(qfluxb(i,j,k)-qfluxt(i,j,k)) 
 100        continue
 110     continue
 120  continue

      return
      end

