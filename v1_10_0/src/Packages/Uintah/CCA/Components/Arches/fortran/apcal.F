c
c Revision 1.4  2000/08/19 16:36:40  rawat
c fixed some bugs in scalarcoef calculations
c
c Revision 1.3  2000/07/28 02:31:04  rawat
c moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
c coeffecients
c
c Revision 1.2  2000/07/12 19:55:45  bbanerje
c Added apcal stuff in calcVelDiagonal
c
c

#include <Packages/Uintah/CCA/Components/Arches/fortran/apcal_fort.h>
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
C
C     APCAL CALCULATES THE DIAGONAL TERM AP OF THE SEVEN-DIAGONAL
C     COEFFICIENT MATRIX BY SUMMING THE OFF-DIAGONAL TERMS (AE,AW,AN,AS,
C     AT, AND AB) AND ADDING THE POSITIVE PART OF THE LINEAR SOURCE 
C     TERMS, SP.  THE CALCULATION IS ONLY DONE FOR THE INTERIOR NODES
C     WHICH ARE SOLVED FOR IN THE CALCULATION, BETWEEN (IST,JST,KST) AND
C     (IEND,JEND,KEND).
*-----------------------------------------------------------------------

*
*     local variables
*
      integer i
      integer j
      integer k
c-----------------------------------------------------------------------
c     executable statements
c-----------------------------------------------------------------------
      do 350 k = valid_lo(3), valid_hi(3)
         do 250 j = valid_lo(2), valid_hi(2)
            do 150 i = valid_lo(1), valid_hi(1)

               ap(i,j,k) = 0.0D0

 150        continue
 250     continue
 350  continue

      do 300 k = valid_lo(3), valid_hi(3)
         do 200 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)

               ap(i,j,k) = ae(i,j,k)+aw(i,j,k)
     $              +an(i,j,k)+as(i,j,k)
     $              +at(i,j,k)+ab(i,j,k)
     $              -sp(i,j,k)

#if 0
               if (ap(i,j,k).lt.0.00000000001) then
                  print*, 'zero ap in ',i,j,k
               endif

               if (isnan_fort(ae(i,j,k))) then
                  print*, 'AE IS NAN AT ',i,j,k
               endif
               if (isnan_fort(aw(i,j,k))) then
                  print*, 'AW IS NAN AT ',i,j,k
               endif
               if (isnan_fort(an(i,j,k))) then
                  print*, 'AN IS NAN AT ',i,j,k
               endif
               if (isnan_fort(as(i,j,k))) then
                  print*, 'AS IS NAN AT ',i,j,k
               endif
               if (isnan_fort(at(i,j,k))) then
                  print*, 'AT IS NAN AT ',i,j,k
               endif
               if (isnan_fort(ab(i,j,k))) then
                  print*, 'AB IS NAN AT ',i,j,k
               endif
               if (isnan_fort(ap(i,j,k))) then
                  print*, 'AP IS NAN AT ',i,j,k
               endif
#endif

 100        continue
 200     continue
 300  continue

      return

      end

c
c $Id$
c
