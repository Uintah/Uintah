c*********************************************************************
c
c
c*********************************************************************

      SUBROUTINE wvelsrc(domLoW, domHiW, domLoWng, domHiWng, 
     $     idxLoW, idxHiW, WW, old_WW,
     &     SU, SP,
     &     domLoU, domHiU, UU,
     &     domLoV, domHiV, VV,
     &     domLoeg, domHieg, domLo, domHi, den, vis,
     $     domLong, domHing, old_den,
     &     grav, deltat, den_ref,
     &     CEE, CWE, CWW, 
     &     CNN, CSN, CSS,
     &     CTTW, CBTW, CBBW, 
     &     SEW, SNS, STBW, STB,
     &     DZPB, 
     &     FAC1W, FAC2W, FAC3W, FAC4W, 
     &     KTSDW, KBSDW,
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
      integer domLoW(3), domHiW(3), idxLoW(3), idxHiW(3)
      integer domLoWng(3), domHiWng(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      double precision old_WW(domLoWng(1):domHiWng(1), 
     $     domLoWng(2):domHiWng(2), 
     &     domLoWng(3):domHiWng(3))
      double precision SU(domLoWng(1):domHiWng(1), 
     $     domLoWng(2):domHiWng(2), 
     &     domLoWng(3):domHiWng(3))
      double precision SP(domLoWng(1):domHiWng(1),
     $     domLoWng(2):domHiWng(2), 
     &     domLoWng(3):domHiWng(3))
      integer domLoU(3), domHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLoeg(3), domHieg(3)
      integer domLo(3), domHi(3)
      double precision DEN(domLoeg(1):domHieg(1), domLoeg(2):domHieg(2), 
     &     domLoeg(3):domHieg(3))
      double precision VIS(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      integer domLong(3), domHing(3)
      double precision old_DEN(domLong(1):domHing(1), 
     $     domLong(2):domHing(2), 
     &     domLong(3):domHing(3))
      double precision DELTAT, GRAV, DEN_REF
      double precision CEE(domLo(1):domHi(1))
      double precision CWE(domLo(1):domHi(1))
      double precision CWW(domLo(1):domHi(1))
      double precision CNN(domLo(2):domHi(2))
      double precision CSN(domLo(2):domHi(2))
      double precision CSS(domLo(2):domHi(2))
      double precision CTTW(domLo(3):domHi(3))
      double precision CBTW(domLo(3):domHi(3))
      double precision CBBW(domLo(3):domHi(3))
      double precision SEW(domLo(1):domHi(1))
      double precision SNS(domLo(2):domHi(2))
      double precision STBW(domLo(3):domHi(3))
      double precision STB(domLo(3):domHi(3))
      double precision DZPB(domLo(3):domHi(3))
      double precision FAC1W(domLo(3):domHi(3))
      double precision FAC2W(domLo(3):domHi(3))
      double precision FAC3W(domLo(3):domHi(3))
      double precision FAC4W(domLo(3):domHi(3))
      integer KTSDW(domLo(3):domHi(3))
      integer KBSDW(domLo(3):domHi(3))
      integer ierr

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
      call ARRASS(SU,ZERO,
     $     domLoWng, domHiWng, idxLoW, idxHiW)
      call ARRASS(SP,ZERO,
     $     domLoWng, domHiWng, idxLoW, idxHiW)

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
     &               (VISNI*DVDZN-VISSI*DVDZS)/SNS(J) +
     &               (VISTI*DWDZT-VISBI*DWDZB)/STBW(K)+
     &               (old_DEN(I,J,K)*(old_WW(I,J,K)-WW(I,J,K))/DELTAT +
     &               (PT5*(DEN(I,J,K-1)+DEN(I,J,K))-DEN_REF)*GRAV))*VOL
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
