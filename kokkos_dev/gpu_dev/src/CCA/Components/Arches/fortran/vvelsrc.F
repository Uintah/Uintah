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

#include <CCA/Components/Arches/fortran/vvelsrc_fort.h>
#include "param4.h"
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTV, JSTV, KSTV, IENDV, JENDV, KENDV
      integer i, j, k

      double precision VOL
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DUDYE, DUDYW, VISEI, VISWI
      double precision DVDYN, DVDYS, VISNI, VISSI
      double precision DWDYB, DWDYT, VISTI, VISBI

c*********************************************************************
c     Start :
c*********************************************************************
      ISTV = idxLoV(1)
      JSTV = idxLoV(2)
      KSTV = idxLoV(3)
      IENDV = idxHiV(1)
      JENDV = idxHiV(2)
      KENDV = idxHiV(3)

c-----------------------------------------------------------------------
c     assemble source coefficients
c-----------------------------------------------------------------------
      call ARRASS(PASS3A(SU)
     & PASS3B(SU),
     & ZERO, idxLoV, idxHiV)
      call ARRASS(PASS3A(SP)
     & PASS3B(SP),
     & ZERO, idxLoV, idxHiV)

c     viscous sources

      do 120 k = KSTV,KENDV
         do 110 j = JSTV,JENDV
            do 100 i = ISTV,IENDV
               VOL = SEW(I)*SNSV(J)*STB(K)
               VISE = PT25*(VIS(I,J,K)+VIS(I+1,J,K)+
     &              VIS(I,J-1,K)+VIS(I+1,J-1,K))
               VISW = PT25*(VIS(I,J,K)+VIS(I-1,J,K)+
     &              VIS(I,J-1,K)+VIS(I-1,J-1,K))
               VISN = FAC1V(J)*VIS(I,J,K)+FAC2V(J)*
     $              VIS(I,J+JNSDV(J),K)
               VISS = FAC3V(J)*VIS(I,J-1,K)+FAC4V(J)*
     $              VIS(I,J+JSSDV(J),K)  
               VIST = PT25*(VIS(I,J,K)+VIS(I,J,K+1)+
     &              VIS(I,J-1,K)+VIS(I,J-1,K+1))
               VISB = PT25*(VIS(I,J,K)+VIS(I,J,K-1)+
     &              VIS(I,J-1,K)+VIS(I,J-1,K-1))
               DUDYE = (UU(I+1,J,K)-UU(I+1,J-1,K))/DYPS(J)
               DUDYW = (UU(I,J,K)-UU(I,J-1,K))/DYPS(J)
               VISEI = CEE(I)*VISE + CWE(I)*VISW
               VISWI = CWW(I)*VISW - CWE(I)*VISE
               DVDYN  = (VV(I,J+1,K)-VV(I,J,K))/SNS(J)
               DVDYS  = (VV(I,J,K)-VV(I,J-1,K))/SNS(J-1)
               VISNI = CNNV(J)*VISN + CSNV(J)*VISS
               VISSI = CSSV(J)*VISS - CSNV(J)*VISN
               DWDYT = (WW(I,J,K+1)-WW(I,J-1,K+1))/DYPS(J)
               DWDYB = (WW(I,J,K)-WW(I,J-1,K))/DYPS(J)
               VISTI = CTT(K)*VIST + CBT(K)*VISB
               VISBI = CBB(K)*VISB - CBT(K)*VIST
               SU(I,J,K) = SU(I,J,K)+
     &              ((VISEI*DUDYE-VISWI*DUDYW)/SEW(I) +
     &              (VISNI*DVDYN-VISSI*DVDYS)/SNSV(J) +
     &              (VISTI*DWDYT-VISBI*DWDYB)/STB(K) +
c     &              old_DEN(I,J,K)*(old_VV(I,J,K)-VV(I,J,K))/
     &              pt5*(old_DEN(I,J,K)+old_DEN(I,J-1,K))*
     $              (old_VV(I,J,K))/
     $              DELTAT
     &              + (PT5*(DEN(I,J-1,K)+DEN(I,J,K))-
     $              pt5*(DEN_REF(i,j,k)+den_ref(i,j-1,k)))*
     $              GRAV
     $              )*VOL
 100        continue
 110     continue
 120  continue
      RETURN
      END

c*********************************************************************
c
c Revision 1.7  2000/10/08 18:56:41  rawat
c fixed the solver for multi
c
c Revision 1.6  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.5  2000/10/06 05:58:09  sparker
c Fixed calls to ARRASS (pass in ng array bounds)
c
c Revision 1.4  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.3  2000/08/10 00:56:38  rawat
c added pressure bc for scalar and changed discretization option for velocity
c
c Revision 1.2  2000/07/28 02:31:04  rawat
c moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
c coeffecients
c
c Revision 1.1  2000/07/12 05:14:27  bbanerje
c Added vvelsrc and wvelsrc .. some changes to uvelsrc.
c Rawat :: Labels are getting hopelessly muddled unless we can do something
c about the time stepping thing.
c
c
c*********************************************************************
