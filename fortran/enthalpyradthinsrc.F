c
c
      subroutine enthalpyradthinsrc(domLo, domHi, idxLo, idxHi,
     $     SU,
     $     domLoT, domHiT, TG,
     $     domLoA, domHiA, Absorp,
     $     domLoG, domHiG,
     $     sew, sns, stb,
     $     TREF)

      implicit none
      INCLUDE 'param4.h'

*     passed variables

      integer domLo(3), domHi(3)
      integer idxLo(3), idxHi(3)
      double precision SU(domLo(1):domHi(1), domLo(2):domHi(2),
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
      double precision sew(domLoG(1):domHiG(1))
      double precision sns(domLoG(2):domHiG(2))
      double precision stb(domLoG(3):domHiG(3))
      double precision tref
      
*     local variables

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision radsource, vol


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
               vol = sew(i)*sns(j)*stb(k)
               radsource = 4*SIGMA*Absorp(i,j,k)*(tg(i,j,k)**4-tref**4)
               su(i,j,k) = su(i,j,k) - radsource*vol
 100        continue
 110     continue
 120  continue

      return
      end

