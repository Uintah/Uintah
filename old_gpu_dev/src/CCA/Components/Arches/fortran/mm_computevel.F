C 
C 
C The MIT License
C 
C Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
C Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
C University of Utah.
C 
C License for the specific language governing rights and limitations under
C Permission is hereby granted, free of charge, to any person obtaining a 
C copy of this software and associated documentation files (the "Software"),
C to deal in the Software without restriction, including without limitation 
C the rights to use, copy, modify, merge, publish, distribute, sublicense, 
C and/or sell copies of the Software, and to permit persons to whom the 
C Software is furnished to do so, subject to the following conditions:
C 
C The above copyright notice and this permission notice shall be included 
C in all copies or substantial portions of the Software.
C 
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
C OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
C THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
C DEALINGS IN THE SOFTWARE.
C 
C
C 
C 
#include <CCA/Components/Arches/fortran/mm_computevel_fort.h>      

c---------------------------------------------------------------------
c     three-dimensional version
c     this subroutine will add pressure gradient term for momentum eqn
c     modification, Seshadri Kumar, July 30, 2001,
c     to take care of multimaterial wall intrusions
c---------------------------------------------------------------------
#include "param4.h"

c*********************************************************************
c     local variables :
c*********************************************************************
      integer i
      integer j
      integer k
      integer idf
      double precision denaverage
      double precision theta

      do 120 k =idxlo_u(3), idxhi_u(3)
         do 110 j = idxlo_u(2), idxhi_u(2)
            do 100 i = idxlo_u(1), idxhi_u(1)

               if ((pcell(i,j,k).ne.mmwallid) .and.
     $              (pcell(i-ioff, j-joff, k-koff)
     $              .ne.mmwallid)) then

                  idf = ioff*i+joff*j+koff*k

                  denaverage = 0.5D0*(den(i-ioff,j-joff,k-koff)+
     +                                den(i,j,k))

                  theta = pt5*(epsg(i-ioff,j-joff,k-koff)+
     $                 epsg(i,j,k))
                  
                  uu(i,j,k) = deltat*theta*
     $                 (press(i-ioff,j-joff,k-koff)-
     $                 press(i,j,k))/dxpw(idf)/denaverage +
     $                 uu(i,j,k)

               else

                  uu(i,j,k) = zero

               endif

 100        continue
 110     continue
 120  continue

      return
      end

