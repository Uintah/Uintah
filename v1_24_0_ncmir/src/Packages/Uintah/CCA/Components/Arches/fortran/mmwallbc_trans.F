c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/mmwallbc_trans_fort.h>
#include "param4.h"
*--------------------------------------------------------------------
*     This subroutine handles intrusion boundary conditions
*          for ETA, F, GETA, GF, HCN, NO, P, PP, and YFU.
*          ALSO NOTE: some of the boundary condition for walls is
*                     handled in the calc routines themselves.  
*                     This happens in TWO places:
*                            1) CP at the boundary
*                            2) GREAT in the intrusion
*--------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i,j,k
      integer ist,jst,kst,iend,jend,kend
      double precision const
      double precision vol      
      double precision apo
      double precision value


c*********************************************************************
c     Start :
c*********************************************************************
      ist = idxLo(1)
      jst = idxLo(2)
      kst = idxLo(3)
      iend = idxHi(1)
      jend = idxHi(2)
      kend = idxHi(3)
      const = one
      DO 500 K = kst,kend
         DO 400 J = jst,jend
            DO 300 I = ist, iend
               IF (PCELL(I,J,K).EQ.MMWALLID) THEN
                  vol = sew(i)*sns(j)*stb(k)
                  value = old_enthalpy(i,j,k)*old_den(i,j,k)*vol/dtime
c intrusion boundary, fix coeffs
                  call fixval_trans(value,ae(i,j,k),aw(i,j,k),an(i,j,k),
     $                        as(i,j,k),at(i,j,k),ab(i,j,k),
     $                        su(i,j,k),sp(i,j,k),const)
               else
                  if (pcell(i-1,j,k).eq.mmwallid)
     $                 aw(i,j,k) = zero
                  if (pcell(i+1,j,k).eq.mmwallid)
     $                 ae(i,j,k) = zero
                  if (pcell(i,j-1,k).eq.mmwallid)
     $                 as(i,j,k) = zero
                  if (pcell(i,j+1,k).eq.mmwallid)
     $                 an(i,j,k) = zero
                  if (pcell(i,j,k-1).eq.mmwallid)
     $                 ab(i,j,k) = zero
                  if (pcell(i,j,k+1).eq.mmwallid)
     $                 at(i,j,k) = zero
               END IF
 300        continue
 400     CONTINUE
 500  CONTINUE
      RETURN
      END 

