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

#include <CCA/Components/Arches/fortran/areain_fort.h>
#include "param4.h"
C---------------------------------------------------------------------
C     THIS SUBROUTINE WILL CALCULATE INLET
C     AREAS FOR THE PRIMARY (AREAPR) AND SECONDARY (AREASC)
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer iBegGhost, jBegGhost, kBegGhost
      integer iEndGhost, jEndGhost, kEndGhost
      integer IST, JST, KST, IEND, JEND, KEND
      integer i, j, k
      double precision AREAT

c*********************************************************************
c     Start :
c*********************************************************************
      iBegGhost = domainLow(1)
      jBegGhost = domainLow(2)
      kBegGhost = domainLow(3)
      iEndGhost = domainHigh(1)
      jEndGhost = domainHigh(2)
      kEndGhost = domainHigh(3)
      IST = indexLow(1)
      JST = indexLow(2)
      KST = indexLow(3)
      IEND = indexHigh(1)
      JEND = indexHigh(2)
      KEND = indexHigh(3)

C-------------------------------------------------------------------- 
C     CALCULATION OF THE AREA OF TOTAL PRIMARY STREAMS
C--------------------------------------------------------------------
      AREAT = ZERO
      areapr = zero

      if (xminus) then
         do  K = KST,KEND
            do  J = JST,JEND
               if (PCELL(iBegGhost,J,K).eq.PFIELD) then
                  if (PCELL(iBegGhost+1,J,K).eq.ffield) then
                    AREAT = AREAT + SNS(J)*STB(K)
                  end if
               end if
            end do
         end do
      end if

      if (xplus) then
         do  K = KST,KEND
            do  J = JST,JEND
               if (PCELL(iEndGhost,J,K).eq.PFIELD) then
                  if (PCELL(iEndGhost-1,J,K).eq.ffield) then
                    AREAT = AREAT + SNS(J)*STB(K)
                  end if
               end if
            end do
         end do
      end if

      if (yminus) then
         do K = KST,KEND
            do I = IST,IEND
               if (PCELL(I,jBegGhost,K).eq.PFIELD) then
                  if (PCELL(I,jBegGhost+1,K).eq.ffield) then
                    AREAT = AREAT + SEW(I)*STB(K)
                  end if
               end if
            end do
         end do
      end if

      if (yplus) then
         do K = KST,KEND
            do I = IST,IEND
               if (PCELL(I,jEndGhost,K).eq.PFIELD) then
                  if (PCELL(I,jEndGhost-1,K).eq.ffield) then
                    AREAT = AREAT + SEW(I)*STB(K)
                  end if
               end if
            end do
         end do
      end if

      if (zminus) then
         do J = JST,JEND
            do I = IST,IEND
               if (PCELL(I,J,kBegGhost).eq.PFIELD) then
                  if (PCELL(I,J,kBegGhost+1).eq.ffield) then
                    AREAT = AREAT + SEW(I)*SNS(J)
                  end if
               end if
            end do
         end do
      end if

      if (zplus) then
         do J = JST,JEND
            do I = IST,IEND
               if (PCELL(I,J,kEndGhost).eq.PFIELD) then
                  if (PCELL(I,J,kEndGhost-1).eq.ffield) then
                    AREAT = AREAT + SEW(I)*SNS(J)
                  end if
               end if
            end do
         end do
      end if

      if (AREAT.eq.ZERO) then
          WRITE(*,*) ' NO PRIMARY INLET AREA IN INPUT GEOMETRY '
      else
          AREAPR = AREAT
      end if

      RETURN
      END
