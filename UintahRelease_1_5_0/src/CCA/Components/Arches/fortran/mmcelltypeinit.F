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
c GENERAL INFORMATION
c 
c    mmcelltypeInit.F
c 
c    Author: Rajesh Rawat (rawat@crsim.utah.edu)
c    University of Utah
c 
c    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
c   
c 
c KEYWORDS
c    Boundary Initialization
c 
c DESCRIPTION
c    Initializes cell type
c   
c WARNING
c    None
c   
c*********************************************************************

#include <CCA/Components/Arches/fortran/mmcelltypeinit_fort.h>
C-----------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     INITIALIZATION ROUTINE     
C-----------------------------------------------------------------------

c*********************************************************************
c     Locals :
c*********************************************************************
      integer i, j, k
      integer iBeg, iEnd
      integer jBeg, jEnd
      integer kBeg, kEnd

      double precision fcutoff

C-----------------------------------------------------------------------
C     Set up the locals
C-----------------------------------------------------------------------
      iBeg = indexLow(1)
      jBeg = indexLow(2)
      kBeg = indexLow(3)
      iEnd = indexHigh(1)
      jEnd = indexHigh(2)
      kEnd = indexHigh(3)

C-----------------------------------------------------------------------
C     RESET all mmcell type to flowtype
C-----------------------------------------------------------------------
      DO 300 K = kBeg,kEnd
         DO 200 J = jBeg,jEnd
            DO 100 I = iBeg,iEnd
               if (PCELL(I,J,K).eq.MMWALLID)
     $              PCELL(I,J,K) = FLOWCELLID

 100        CONTINUE
 200     CONTINUE
 300  CONTINUE

      DO 310 K = kBeg,kEnd
         DO 210 J = jBeg,jEnd
            DO 110 I = iBeg,iEnd

               if (VOIDFRAC(i,j,k).le.cutoff) then

                  PCELL(I,J,K) = MMWALLID
                  VOIDFRAC(I,J,K) = 0.0D0

               else

                  voidfrac(i,j,k) = 1.0d0

               endif

 110        CONTINUE
 210     CONTINUE
 310  CONTINUE

      RETURN
      END

