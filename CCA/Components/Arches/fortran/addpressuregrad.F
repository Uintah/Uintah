c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE addpressuregrad(domLoU, domHiU, domLoUng, domHiUng,
     $     idxLoU, idxHiU, gradP, 
     &     SU,
     &     domLo, domHi, pcell, mmwallid,
     &     ioff, joff, koff,
     &     ierr)
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL add pressure gradient term for momentum eqn
C---------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3), domLoUng(3), domHiUng(3)
      integer idxLoU(3), idxHiU(3)
      double precision gradP(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision SU(domLoUng(1):domHiUng(1), 
     $     domLoUng(2):domHiUng(2), 
     &     domLoUng(3):domHiUng(3))
      integer domLo(3), domHi(3)
      integer PCELL(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer ioff, joff, koff
      integer mmwallid

      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i, j, k

      do 120 k =idxLoU(3), idxHiU(3)
         do 110 j = idxLoU(2), idxHiU(2)
            do 100 i = idxLoU(1), idxHiU(1)
               if (pcell(i,j,k).ne.mmwallid.and.
     &                 pcell(i-ioff,j-joff,k-koff).ne.mmwallid) then
                  su(i,j,k) = su(i,j,k)+ gradP(i,j,k)
               end if
 100        continue
 110     continue
 120  continue
      RETURN
      END

c*********************************************************************
c
c Revision 1.5  2000/10/09 17:06:30  rawat
c modified momentum solver for multi-patch
c
c Revision 1.4  2000/09/20 16:56:19  rawat
c added some petsc parallel stuff and fixed some bugs
c
c Revision 1.3  2000/08/15 00:23:38  rawat
c added explicit solve for momentum and scalar eqns
c
c Revision 1.2  2000/07/28 02:31:04  rawat
c moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
c coeffecients
c
c Revision 1.1  2000/07/17 22:07:02  rawat
c modified momentum source
c
c
c*********************************************************************
