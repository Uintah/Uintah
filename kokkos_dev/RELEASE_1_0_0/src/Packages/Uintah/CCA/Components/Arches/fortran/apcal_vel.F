c
c Revision 1.1  2000/08/19 16:36:40  rawat
c fixed some bugs in scalarcoef calculations
c
c Revision 1.3  2000/07/28 02:31:04  rawat
c moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store
c matrix
c coeffecients
c
c Revision 1.2  2000/07/12 19:55:45  bbanerje
c Added apcal stuff in calcVelDiagonal
c
c

      SUBROUTINE apcalvel(dim_lo, dim_hi, valid_lo, valid_hi,
     &     AP,AE,AW,AN,AS,AT,AB,SP)
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
      implicit none

      integer dim_lo(3)
      integer dim_hi(3)
      integer valid_lo(3)
      integer valid_hi(3)
*
*     passed variables
*
      double precision ap(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision ae(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision aw(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision an(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision as(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision at(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision ab(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

      double precision sp(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))

*
*     local variables
*
      integer i
      integer j
      integer k
c-----------------------------------------------------------------------
c     executable statements
c-----------------------------------------------------------------------
      do 350 k = dim_lo(3), dim_hi(3)
         do 250 j = dim_lo(2), dim_hi(2)
            do 150 i = dim_lo(1), dim_hi(1)

               ap(i,j,k) = 1.0

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

 100        continue
 200     continue
 300  continue

      return

      end

c
c $Id$
c
