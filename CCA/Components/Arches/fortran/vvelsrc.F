c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE vvelsrc(
     $     domLoV, domHiV, domLoVng, domHiVng,
     $     idxLoV, idxHiV, VV, old_VV,
     &     SU, SP,
     &     domLoU, domHiU, UU,
     &     domLoW, domHiW, WW,
     &     domLoeg, domHieg, domLo, domHi, den, vis,
     $     domLong, domHing, old_den,
     $     domLodenref, domHidenref, den_ref,
     &     grav, deltat, 
     &     CEE, CWE, CWW, 
     &     CNNV, CSNV, CSSV,
     &     CTT, CBT, CBB, 
     &     SEW, SNSV, SNS, STB,
     &     DYPS, 
     &     FAC1V, FAC2V, FAC3V, FAC4V, 
     &     JNSDV, JSSDV,
     &     ierr)
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoV(3), domHiV(3), idxLoV(3), idxHiV(3)
      integer domLoVng(3), domHiVng(3)

      double precision VV(
     $     domLoV(1):domHiV(1), 
     $     domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))

      double precision old_VV(
     $     domLoVng(1):domHiVng(1), 
     $     domLoVng(2):domHiVng(2), 
     &     domLoVng(3):domHiVng(3))

      double precision SU(
     $     domLoVng(1):domHiVng(1), 
     $     domLoVng(2):domHiVng(2), 
     &     domLoVng(3):domHiVng(3))

      double precision SP(
     $     domLoVng(1):domHiVng(1),
     $     domLoVng(2):domHiVng(2), 
     &     domLoVng(3):domHiVng(3))

      integer domLoU(3), domHiU(3)

      double precision UU(
     $     domLoU(1):domHiU(1), 
     $     domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))

      integer domLoW(3), domHiW(3)

      double precision WW(
     $     domLoW(1):domHiW(1), 
     $     domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))

      integer domLoeg(3), domHieg(3)
      integer domLo(3), domHi(3)

      double precision DEN(
     $     domLoeg(1):domHieg(1), 
     $     domLoeg(2):domHieg(2), 
     &     domLoeg(3):domHieg(3))

      double precision VIS(
     $     domLo(1):domHi(1), 
     $     domLo(2):domHi(2), 
     &     domLo(3):domHi(3))

      integer domLong(3), domHing(3)

      double precision old_DEN(
     $     domLong(1):domHing(1), 
     $     domLong(2):domHing(2), 
     &     domLong(3):domHing(3))

      integer domLodenref(3), domHidenref(3)

      double precision DEN_REF(
     $     domLodenref(1):domHidenref(1), 
     $     domLodenref(2):domHidenref(2), 
     &     domLodenref(3):domHidenref(3))

      double precision DELTAT, GRAV

      double precision CEE(domLo(1):domHi(1))
      double precision CWE(domLo(1):domHi(1))
      double precision CWW(domLo(1):domHi(1))
      double precision CNNV(domLo(2):domHi(2))
      double precision CSNV(domLo(2):domHi(2))
      double precision CSSV(domLo(2):domHi(2))
      double precision CTT(domLo(3):domHi(3))
      double precision CBT(domLo(3):domHi(3))
      double precision CBB(domLo(3):domHi(3))
      double precision SEW(domLo(1):domHi(1))
      double precision SNS(domLo(2):domHi(2))
      double precision SNSV(domLo(2):domHi(2))
      double precision STB(domLo(3):domHi(3))
      double precision DYPS(domLo(2):domHi(2))
      double precision FAC1V(domLo(2):domHi(2))
      double precision FAC2V(domLo(2):domHi(2))
      double precision FAC3V(domLo(2):domHi(2))
      double precision FAC4V(domLo(2):domHi(2))
      integer JNSDV(domLo(2):domHi(2))
      integer JSSDV(domLo(2):domHi(2))
      integer ierr

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
      call ARRASS(SU,ZERO,
     $     domLoVng, domHiVng, idxLoV, idxHiV)
      call ARRASS(SP,ZERO,
     $     domLoVng, domHiVng, idxLoV, idxHiV)

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
