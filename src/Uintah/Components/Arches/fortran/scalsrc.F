c
c $Id$
c
      subroutine scalsrc(domLo, domHi, idxLo, idxHi, 
     $     sp, su, 
     &     old_den, old_scal, 
     &     sew, sns, stb, 
     &     deltat)

      implicit none
      INCLUDE 'param4.h'

*     passed variables

      integer domLo(3), domHi(3)
      integer idxLo(3), idxHi(3)
      double precision su(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision sp(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision old_den(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
      double precision old_scal(domLo(1):domHi(1), domLo(2):domHi(2),
     $     domLo(3):domHi(3))
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

      CALL ARRASS(SU,ZERO,domLo, domHi, idxLo, idxHi)
      CALL ARRASS(SP,ZERO,domLo, domHi, idxLo, idxHi)

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
c $Log$
c Revision 1.2  2000/08/19 16:36:40  rawat
c fixed some bugs in scalarcoef calculations
c
c Revision 1.1  2000/07/30 22:59:32  bbanerje
c Added scalar source term calcs.
c
c
