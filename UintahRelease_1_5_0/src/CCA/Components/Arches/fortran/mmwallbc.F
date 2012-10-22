C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/mmwallbc_fort.h>
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

#undef wallmodel
#define wallmodel

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i,j,k
      integer ist,jst,kst,iend,jend,kend
      double precision const

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
c intrusion boundary, fix coeffs
                  call fixval(zero,ae(i,j,k),aw(i,j,k),an(i,j,k),
     $                        as(i,j,k),at(i,j,k),ab(i,j,k),
     $                        su(i,j,k),sp(i,j,k),const)
               else
#ifdef wallmodel
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
#endif
               END IF
 300        continue
 400     CONTINUE
 500  CONTINUE
      RETURN
      END 

