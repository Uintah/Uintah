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
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/wallbc_fort.h>
#include "param4.h"

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer i,j,k
      integer ist,jst,kst,iend,jend,kend

c*********************************************************************
c     Start :
c*********************************************************************
      ist = idxLo(1)
      jst = idxLo(2)
      kst = idxLo(3)
      iend = idxHi(1)
      jend = idxHi(2)
      kend = idxHi(3)

c WARNING! Only xminus wall BC is supported
      if (xminus) then
         DO 500 K = kst,kend
            DO 400 J = jst,jend
               I = ist -1
               IF (PCELL(I,J,K).EQ.WALL) THEN
                  IF (PCELL(I+1,J,K).NE.WALL) 
     $                 AW(I+1,J,K) = ZERO
               END IF
 400        CONTINUE
 500     CONTINUE
      endif
      RETURN
      END 
