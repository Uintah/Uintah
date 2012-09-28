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
c
c
#include <CCA/Components/Arches/fortran/enthalpyradthinsrc_fort.h>
#include "param4.h"
      
*     local variables

      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision radsource, vol


      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)
c-----------------------------------------------------------------------

      do 120 k = KST, KEND
         do 110 j = JST, JEND
            do 100 i = IST, IEND
               vol = sew(i)*sns(j)*stb(k)
               radsource = 4.0D0*SIGMA*Absorp(i,j,k)*
     *(tg(i,j,k)**4-tref**4)
               su(i,j,k) = su(i,j,k) - radsource*vol
 100        continue
 110     continue
 120  continue

      return
      end

