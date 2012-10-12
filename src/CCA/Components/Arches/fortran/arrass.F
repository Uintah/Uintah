C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the \"Software\"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C
C 
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/arrass_fort.h>
C-----------------------------------------------------------------------
C     WRITTEN, OCTOBER 14, 1996, BY SESHADRI KUMAR
C
C     ASSIGNS A CONSTANT ALPHA TO ALL ELEMENTS OF AN ARRAY AA, i.e.,
C
C     AA = ALPHA
C
C-----------------------------------------------------------------------

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
c Revision 1.2  2000/07/11 15:46:29  rawat
c added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
c
c
c*********************************************************************
