c
c
      subroutine scalsrc(domLo, domHi, domLong, domHing, idxLo, idxHi, 
     $     sp, su, 
     &     old_den, old_scal, 
     &     sew, sns, stb, 
     &     deltat)

      implicit none
      INCLUDE 'param4.h'

*     passed variables

      integer domLo(3), domHi(3)
      integer domLong(3), domHing(3)
      integer idxLo(3), idxHi(3)
      double precision su(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision sp(domLong(1):domHing(1), domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision old_den(domLong(1):domHing(1), 
     $     domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision old_scal(domLong(1):domHing(1), 
     $     domLong(2):domHing(2),
     $     domLong(3):domHing(3))
      double precision sew(domLo(1):domHi(1))
      double precision sns(domLo(2):domHi(2))
      double precision stb(domLo(3):domHi(3))
      double precision deltat
      
*     local variables

      double precision vol, apo
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      logical ltim, lexpl

      ltim = .true.
      lexpl = .true.
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------

      CALL ARRASS(SU,ZERO,domLong, domHing, idxLo, idxHi)
      CALL ARRASS(SP,ZERO,domLong, domHing, idxLo, idxHi)

      do 120 k = KST, KEND
         do 110 j = JST, JEND
            do 100 i = IST, IEND
               vol = sns(j)*sew(i)*stb(k)
               IF (LTIM) THEN
                  APO = old_den(I,J,K)*VOL/deltat
                  SU(I,J,K) = SU(I,J,K) + APO*old_scal(I,J,K)
                  IF (.NOT.LEXPL) SP(I,J,K) = SP(I,J,K) - APO
               END IF
 100        continue
 110     continue
 120  continue

      return
      end
c
c Revision 1.3  2000/10/10 19:31:03  rawat
c added scalarsolver
c
c Revision 1.2  2000/08/19 16:36:40  rawat
c fixed some bugs in scalarcoef calculations
c
c Revision 1.1  2000/07/30 22:59:32  bbanerje
c Added scalar source term calcs.
c
c
