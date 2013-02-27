C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/uvelsrc_fort.h>
#include "param4.h"
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer i, j, k

      double precision VOL
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DUDXE, DUDXW, VISEI, VISWI
      double precision DVDXN, DVDXS, VISNI, VISSI
      double precision DWDXB, DWDXT, VISTI, VISBI

c*********************************************************************
c     Start :
c*********************************************************************
      ISTU = idxLoU(1)
      JSTU = idxLoU(2)
      KSTU = idxLoU(3)
      IENDU = idxHiU(1)
      JENDU = idxHiU(2)
      KENDU = idxHiU(3)

c-----------------------------------------------------------------------
c     assemble source coefficients
c-----------------------------------------------------------------------
      call ARRASS(PASS3A(SU)
     & PASS3B(SU),
     & ZERO,idxLoU, idxHiU)
      call ARRASS(PASS3A(SP)
     & PASS3B(SP),
     & ZERO,idxLoU, idxHiU)

c     viscous sources

      do 120 k = KSTU,KENDU
         do 110 j = JSTU,JENDU
            do 100 i = ISTU,IENDU
               VOL = SEWU(I)*SNS(J)*STB(K)
               VISE = FAC1U(I)*VIS(I,J,K)+FAC2U(I)*VIS(I+IESDU(I),J,K)
               VISW = FAC3U(I)*VIS(I-1,J,K)+FAC4U(I)*
     $              VIS(I+IWSDU(I),J,K)
               VISN = PT25*(VIS(I,J,K)+VIS(I,J+1,K)+
     &              VIS(I-1,J,K)+VIS(I-1,J+1,K))
               VISS = PT25*(VIS(I,J,K)+VIS(I,J-1,K)+
     &              VIS(I-1,J,K)+VIS(I-1,J-1,K))
               VIST = PT25*(VIS(I,J,K)+VIS(I,J,K+1)+
     &              VIS(I-1,J,K)+VIS(I-1,J,K+1))
               VISB = PT25*(VIS(I,J,K)+VIS(I,J,K-1)+
     &              VIS(I-1,J,K)+VIS(I-1,J,K-1))  
               DUDXE  = (UU(I+1,J,K)-UU(I,J,K))/SEW(I)
               DUDXW  = (UU(I,J,K)-UU(I-1,J,K))/SEW(I-1)
               VISEI = CEEU(I)*VISE + CWEU(I)*VISW
               VISWI = CWWU(I)*VISW - CWEU(I)*VISE
               DVDXN  = (VV(I,J+1,K)-VV(I-1,J+1,K))/DXPW(I)
               DVDXS  = (VV(I,J,K)-VV(I-1,J,K))/DXPW(I)
               VISNI = CNN(J)*VISN + CSN(J)*VISS
               VISSI = CSS(J)*VISS - CSN(J)*VISN
               DWDXT = (WW(I,J,K+1)-WW(I-1,J,K+1))/DXPW(I)
               DWDXB = (WW(I,J,K)-WW(I-1,J,K))/DXPW(I)
               VISTI = CTT(K)*VIST + CBT(K)*VISB
               VISBI = CBB(K)*VISB - CBT(K)*VIST
c
c   Add all the source terms at one go (don't need as many array searches)
c
c modifying projection for explicit methods
               SU(I,J,K) = SU(I,J,K)+
     &              ((VISEI*DUDXE-VISWI*DUDXW)/SEWU(I) +
     &              (VISNI*DVDXN-VISSI*DVDXS)/SNS(J)   +
     &              (VISTI*DWDXT-VISBI*DWDXB)/STB(K)   +
c     &              old_DEN(I,J,K)*(old_UU(I,J,K)-UU(I,J,K))/
     &              pt5*(old_DEN(I,J,K)+old_DEN(I-1,J,K))*
     $              (old_UU(I,J,K))/
     $              DELTAT
     &              + (PT5*(DEN(I,J,K)+DEN(I-1,J,K))-
     $              pt5*(den_ref(i,j,k)+den_ref(i-1,j,k)))*
     $              GRAV
     $              )*VOL 

#if 0
               temp = (PT5*(DEN(I,J,K)+DEN(I-1,J,K))-
     $              pt5*(den_ref(i,j,k)+den_ref(i-1,j,k)))*
     $              GRAV*VOL 

               temp2 = pt5*(old_DEN(I,J,K)+old_DEN(I-1,J,K))*
     $              (old_UU(I,J,K))/
     $              DELTAT

               temp3 = ((VISEI*DUDXE-VISWI*DUDXW)/SEWU(I) +
     &              (VISNI*DVDXN-VISSI*DVDXS)/SNS(J)   +
     &              (VISTI*DWDXT-VISBI*DWDXB)/STB(K))*VOL
#endif

 100        continue
 110     continue
 120  continue
      RETURN
      END

c*********************************************************************
c
c Revision 1.10  2000/10/08 18:56:41  rawat
c fixed the solver for multi
c
c Revision 1.9  2000/10/06 23:07:53  rawat
c fixed some more bc routines for mulit-patch
c
c Revision 1.8  2000/10/06 05:58:09  sparker
c Fixed calls to ARRASS (pass in ng array bounds)
c
c Revision 1.7  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.6  2000/08/23 06:20:53  bbanerje
c 1) Results now correct for pressure solve.
c 2) Modified BCU, BCV, BCW to add stuff for pressure BC.
c 3) Removed some bugs in BCU, V, W.
c 4) Coefficients for MOM Solve not computed correctly yet.
c
c Revision 1.5  2000/08/10 00:56:38  rawat
c added pressure bc for scalar and changed discretization option for velocity
c
c Revision 1.4  2000/07/28 02:31:04  rawat
c moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
c coeffecients
c
c Revision 1.3  2000/07/12 05:14:27  bbanerje
c Added vvelsrc and wvelsrc .. some changes to uvelsrc.
c Rawat :: Labels are getting hopelessly muddled unless we can do something
c about the time stepping thing.
c
c Revision 1.2  2000/07/11 15:46:30  rawat
c added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
c
c Revision 1.1  2000/07/09 00:23:06  bbanerje
c Dummy version of uvelsrc copied from ucoef.
c
c
c*********************************************************************
