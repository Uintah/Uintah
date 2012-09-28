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
#include <CCA/Components/Models/Radiation/fortran/m_rdomflux_fort.h>
      integer i,j,k

      do 30 k=idxlo(3),idxhi(3)
         do 20 j=idxlo(2),idxhi(2)
            do 10 i=idxlo(1),idxhi(1)

               if (plusX) then
                  qfluxe(i,j,k) = qfluxe(i,j,k) + wt(l) 
     1                 * abs(omu(l)) * cenint(i,j,k)
               else
                  qfluxw(i,j,k) = qfluxw(i,j,k) + wt(l) 
     1                 * abs(omu(l)) * cenint(i,j,k)
               endif

               if (plusY) then
                  qfluxn(i,j,k) = qfluxn(i,j,k) + wt(l) 
     1                 * abs(oeta(l)) * cenint(i,j,k)
               else
                  qfluxs(i,j,k) = qfluxs(i,j,k) + wt(l) 
     1                 * abs(oeta(l)) * cenint(i,j,k)
               endif

               if (plusZ) then
                  qfluxt(i,j,k) = qfluxt(i,j,k) + wt(l) 
     1                 * abs(oxi(l)) * cenint(i,j,k)
               else
                  qfluxb(i,j,k) = qfluxb(i,j,k) + wt(l) 
     1                 * abs(oxi(l)) * cenint(i,j,k)
               endif

 10         continue
 20      continue
 30   continue

      RETURN
      end





