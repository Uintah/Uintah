# 1 "arrass.F"
c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE arrass(
     $     AA,alpha,
     &     dim_lo, dim_hi, valid_lo, valid_hi)
C-----------------------------------------------------------------------
C     WRITTEN, OCTOBER 14, 1996, BY SESHADRI KUMAR
C
C     ASSIGNS A CONSTANT ALPHA TO ALL ELEMENTS OF AN ARRAY AA, i.e.,
C
C     AA = ALPHA
C
C-----------------------------------------------------------------------
      implicit none

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer dim_lo(3)
c
c     lower indices for dimension of AA
c
      integer dim_hi(3)
c
c     upper indices for dimension of AA
c
      integer valid_lo(3)
c
c     lower indices of valid region of AA over which operation is
c     to be performed
c
      integer valid_hi(3)
c
c     upper indices of valid region of AA over which operation is
c     to be performed
c
      double precision AA(
     $     dim_lo(1):dim_hi(1),
     $     dim_lo(2):dim_hi(2),
     $     dim_lo(3):dim_hi(3))
c     
c     input array, modified as output
c
      double precision ALPHA
c     
c     constant scalar multiple
c
c*********************************************************************
c     Local Variables :
c*********************************************************************

      integer i,j,k

c*********************************************************************
c     Start :
c*********************************************************************

C-----------------------------------------------------------------------
C     EXECUTABLE STATEMENTS
C-----------------------------------------------------------------------

      do 300 k = valid_lo(3), valid_hi(3)
         do 200 j = valid_lo(2), valid_hi(2)
            do 100 i = valid_lo(1), valid_hi(1)
               AA(I,J,K) = ALPHA
 100        continue
 200     continue
 300  continue

      return
      end

c*********************************************************************
c
c $Log$
c Revision 1.1  2001/04/04 15:44:22  rawat
c modified MPMArches from the old tree written by Kumar to work in the new SCIRun
c added all the fortran subroutines required by MPMArches
c
c
c*********************************************************************
