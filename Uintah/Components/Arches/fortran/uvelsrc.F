c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE uvelsrc(domLoU, domHiU, idxLoU, idxHiU, UU, old_UU,
     &     SU, SP,
     &     domLoV, domHiV, VV,
     &     domLoW, domHiW, WW,
     &     domLo, domHi, den, old_den, vis,
     &     grav, deltat, den_ref,
     &     CEEU, CWEU, CWWU, 
     &     CNN, CSN, CSS,
     &     CTT, CBT, CBB, 
     &     SEWU, SEW, SNS, STB,
     &     DXPW, 
     &     FAC1U, FAC2U, FAC3U, FAC4U, 
     &     IESDU, IWSDU,
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
      integer domLoU(3), domHiU(3), idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision old_UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision SU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision SP(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      integer domLoV(3), domHiV(3)
      double precision VV(domLoV(1):domHiV(1), domLoV(2):domHiV(2), 
     &     domLoV(3):domHiV(3))
      integer domLoW(3), domHiW(3)
      double precision WW(domLoW(1):domHiW(1), domLoW(2):domHiW(2), 
     &     domLoW(3):domHiW(3))
      integer domLo(3), domHi(3)
      double precision DEN(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision old_DEN(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision VIS(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision DELTAT, GRAV, DEN_REF
      double precision CEEU(domLo(1):domHi(1))
      double precision CWEU(domLo(1):domHi(1))
      double precision CWWU(domLo(1):domHi(1))
      double precision CNN(domLo(2):domHi(2))
      double precision CSN(domLo(2):domHi(2))
      double precision CSS(domLo(2):domHi(2))
      double precision CTT(domLo(3):domHi(3))
      double precision CBT(domLo(3):domHi(3))
      double precision CBB(domLo(3):domHi(3))
      double precision SEWU(domLo(1):domHi(1))
      double precision SEW(domLo(1):domHi(1))
      double precision SNS(domLo(2):domHi(2))
      double precision STB(domLo(3):domHi(3))
      double precision DXPW(domLo(1):domHi(1))
      double precision FAC1U(domLo(1):domHi(1))
      double precision FAC2U(domLo(1):domHi(1))
      double precision FAC3U(domLo(1):domHi(1))
      double precision FAC4U(domLo(1):domHi(1))
      integer IESDU(domLo(1):domHi(1))
      integer IWSDU(domLo(1):domHi(1))
      integer ierr

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
      call ARRASS(SU,ZERO,
     $     domLoU, domHiU, idxLoU, idxHiU)
      call ARRASS(SP,ZERO,
     $     domLoU, domHiU, idxLoU, idxHiU)

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
               SU(I,J,K) = SU(I,J,K)+
     &              ((VISEI*DUDXE-VISWI*DUDXW)/SEWU(I) +
     &              (VISNI*DVDXN-VISSI*DVDXS)/SNS(J)   +
     &              (VISTI*DWDXT-VISBI*DWDXB)/STB(K)   +
     &              (old_DEN(I,J,K)*(old_UU(I,J,K)-UU(I,J,K))/DELTAT +
     &              (PT5*(DEN(I,J,K)+DEN(I-1,J,K))-DEN_REF)*GRAV))*VOL 


 100        continue
 110     continue
 120  continue
      RETURN
      END

c*********************************************************************
c
c $Log$
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
