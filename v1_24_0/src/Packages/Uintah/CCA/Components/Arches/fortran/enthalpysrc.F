c
c
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpysrc_fort.h>
      
*     local variables

      double precision vol, apo
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      logical*1 ltim, lexpl

      ltim = .true.
      lexpl = .true.
      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------

      CALL ARRASS(PASS3A(SU)
     & PASS3B(SU),
     & ZERO, idxLo, idxHi)
      CALL ARRASS(PASS3A(SP)
     & PASS3B(SU),
     & ZERO, idxLo, idxHi)

      do 120 k = KST, KEND
         do 110 j = JST, JEND
            do 100 i = IST, IEND
               vol = sns(j)*sew(i)*stb(k)
               IF (LTIM) THEN
                  APO = old_den(I,J,K)*VOL/deltat
                  SU(I,J,K) = SU(I,J,K) + APO*old_enth(I,J,K)
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
