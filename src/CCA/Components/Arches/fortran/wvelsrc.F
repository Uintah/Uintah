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

#include <CCA/Components/Arches/fortran/wvelsrc_fort.h>
#include "param4.h"
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTW, JSTW, KSTW, IENDW, JENDW, KENDW
      integer i, j, k

      double precision VOL
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DUDZE, DUDZW, VISEI, VISWI
      double precision DVDZN, DVDZS, VISNI, VISSI
      double precision DWDZB, DWDZT, VISTI, VISBI

c*********************************************************************
c     Start :
c*********************************************************************
      ISTW = idxLoW(1)
      JSTW = idxLoW(2)
      KSTW = idxLoW(3)
      IENDW = idxHiW(1)
      JENDW = idxHiW(2)
      KENDW = idxHiW(3)

c-----------------------------------------------------------------------
c     assemble source coefficients
c-----------------------------------------------------------------------
      call ARRASS(PASS3A(SU)
     & PASS3B(SU),
     & ZERO, idxLoW, idxHiW)
      call ARRASS(PASS3A(SP)
     & PASS3B(SP),
     & ZERO, idxLoW, idxHiW)

c     viscous sources

      do 120 k = KSTW,KENDW
         do 110 j = JSTW,JENDW
            do 100 i = ISTW,IENDW
               VOL = SEW(I)*SNS(J)*STBW(K)
               VISE = PT25*(VIS(I,J,K)+VIS(I+1,J,K)+
     &              VIS(I,J,K-1)+VIS(I+1,J,K-1))
               VISW = PT25*(VIS(I,J,K)+VIS(I-1,J,K)+
     &              VIS(I,J,K-1)+VIS(I-1,J,K-1))
               VISN = PT25*(VIS(I,J,K)+VIS(I,J+1,K)+
     &              VIS(I,J,K-1)+VIS(I,J+1,K-1))
               VISS = PT25*(VIS(I,J,K)+VIS(I,J-1,K)+
     &              VIS(I,J,K-1)+VIS(I,J-1,K-1))
               VIST = FAC1W(K)*VIS(I,J,K)+FAC2W(K)*
     &              VIS(I,J,K+KTSDW(K))
               VISB = FAC3W(K)*VIS(I,J,K-1)+FAC4W(K)*
     &              VIS(I,J,K+KBSDW(K))
               DUDZE = (UU(I+1,J,K)-UU(I+1,J,K-1))/DZPB(K)
               DUDZW = (UU(I,J,K)-UU(I,J,K-1))/DZPB(K)
               VISEI = CEE(I)*VISE + CWE(I)*VISW
               VISWI = CWW(I)*VISW - CWE(I)*VISE
               DVDZN = (VV(I,J+1,K)-VV(I,J+1,K-1))/DZPB(K)
               DVDZS = (VV(I,J,K)-VV(I,J,K-1))/DZPB(K)
               VISNI = CNN(J)*VISN + CSN(J)*VISS
               VISSI = CSS(J)*VISS - CSN(J)*VISN
               DWDZT = (WW(I,J,K+1)-WW(I,J,K))/STB(K)
               DWDZB = (WW(I,J,K)-WW(I,J,K-1))/STB(K-1)
               VISTI = CTTW(K)*VIST + CBTW(K)*VISB
               VISBI = CBBW(K)*VISB - CBTW(K)*VIST
               SU(I,J,K) = SU(I,J,K)+
     &              ((VISEI*DUDZE-VISWI*DUDZW)/SEW(I) +
     &              (VISNI*DVDZN-VISSI*DVDZS)/SNS(J) +
     &              (VISTI*DWDZT-VISBI*DWDZB)/STBW(K)+
c     &              old_DEN(I,J,K)*(old_WW(I,J,K)-WW(I,J,K))/
     &              pt5*(old_DEN(I,J,K)+old_DEN(i,j,k-1))*
     $              (old_WW(I,J,K))/
     $              DELTAT 
     &              + (PT5*(DEN(I,J,K-1)+DEN(I,J,K))-
     $              pt5*(den_ref(i,j,k) + den_ref(i,j,k-1)))*
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
c Revision 1.3  2000/08/10 21:29:15  rawat
c fixed a bug in cellinformation
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
