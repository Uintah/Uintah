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
     &     CEEU, CWEU, CWWU, CNN, CSN, CSS,
     &     CTT, CBT, CBB, SEWU, SEW, SNS, STB,
     &     DXEPU, DXPWU, DXPW, DYNP, DYPS, DZTP, DZPB, 
     &     FAC1U, FAC2U, FAC3U, FAC4U, 
     &     IESDU, IWSDU,
     &     ENFAC, SFAC, TFAC, BFAC,
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
      double precision DELTAT, grav, den_ref
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
      double precision DXEPU(domLo(1):domHi(1))
      double precision DXPWU(domLo(1):domHi(1))
      double precision DXPW(domLo(1):domHi(1))
      double precision DYNP(domLo(2):domHi(2))
      double precision DYPS(domLo(2):domHi(2))
      double precision DZTP(domLo(3):domHi(3))
      double precision DZPB(domLo(3):domHi(3))
      double precision FAC1U(domLo(1):domHi(1))
      double precision FAC2U(domLo(1):domHi(1))
      double precision FAC3U(domLo(1):domHi(1))
      double precision FAC4U(domLo(1):domHi(1))
      integer IESDU(domLo(1):domHi(1))
      integer IWSDU(domLo(1):domHi(1))
      double precision ENFAC(domLo(2):domHi(2))
      double precision SFAC(domLo(2):domHi(2))
      double precision TFAC(domLo(3):domHi(3))
      double precision BFAC(domLo(3):domHi(3))
      integer ierr

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer i, j, k

      logical L2UP, LSMPT, LCEND, LTEST, LTIM

      double precision AREAN, AREAS, AREAEW, AREATB, VOL
      double precision GE, GP, GW, GN, GNW, GS, GSW, GT, GTW, GTP, GBW
      double precision CEO, CWO, CNO, CSO, CTO, CBO
      double precision CE, CW, CN, CS, CT, CB
      double precision CPE, CPW, GVP, CPN, CPS, GWP, CPT, CPB
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DEO, DWO, DNO, DSO, DTO, DBO
      double precision DE, DW, DN, DS, DT, DB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision TEW, TNS, TTB
      double precision AEC, AWC, ANC, ASC, ATC, ABC
      double precision dudxe, dudxw, visei, viswi
      double precision dvdxn, dvdxs, visni, vissi
      double precision dwdxb, dwdxt, visti, visbi

c*********************************************************************
c     Start :
c*********************************************************************
      ISTU = idxLoU(1)
      JSTU = idxLoU(2)
      KSTU = idxLoU(3)
      IENDU = idxHiU(1)
      JENDU = idxHiU(2)
      KENDU = idxHiU(3)

c*********************************************************************
c     Set all flags except one to false (** WARNING ** need to change)
c*********************************************************************
      L2UP = .TRUE.
      LSMPT = .FALSE.
      LCEND = .FALSE.
      LTEST = .FALSE.
      LTIM = .TRUE.

c-----------------------------------------------------------------------
c     assemble source coefficients
c-----------------------------------------------------------------------
      VOL = SEWU(I)*SNS(J)*STB(K)
      call arrass(su,zero,
     $     domLoU, domHiU, idxLoU, idxHiU)
      call arrass(sp,zero,
     $     domLoU, domHiU, idxLoU, idxHiU)

c     viscous sources

      do 120 k=idxLoU(3),idxHiU(3)
         do 110 j=idxLoU(2),idxHiU(2)
            do 100 i=idxLoU(1),idxHiU(1)
               vise = fac1u(i)*vis(i,j,k)+fac2u(i)*vis(i+iesdu(i),j,k)
               visw = fac3u(i)*vis(i-1,j,k)+fac4u(i)*
     $              vis(i+iwsdu(i),j,k)
               visn = pt25*(vis(i,j,k)+vis(i,j+1,k)+
     &              vis(i-1,j,k)+vis(i-1,j+1,k))
               viss = pt25*(vis(i,j,k)+vis(i,j-1,k)+
     &              vis(i-1,j,k)+vis(i-1,j-1,k))
               vist = pt25*(vis(i,j,k)+vis(i,j,k+1)+
     &              vis(i-1,j,k)+vis(i-1,j,k+1))
               visb = pt25*(vis(i,j,k)+vis(i,j,k-1)+
     &              vis(i-1,j,k)+vis(i-1,j,k-1))  
               dudxe  = (uu(i+1,j,k)-uu(i,j,k))/sew(i)
               dudxw  = (uu(i,j,k)-uu(i-1,j,k))/sew(i-1)
               visei = ceeu(i)*vise + cweu(i)*visw
               viswi = cwwu(i)*visw - cweu(i)*vise
               su(i,j,k) = su(i,j,k)+(visei*dudxe-viswi*dudxw)/
     &              sewu(i)*vol
               dvdxn  = (vv(i,j+1,k)-vv(i-1,j+1,k))/dxpw(i)
               dvdxs  = (vv(i,j,k)-vv(i-1,j,k))/dxpw(i)
               visni = cnn(j)*visn + csn(j)*viss
               vissi = css(j)*viss - csn(j)*visn
               su(i,j,k) = su(i,j,k)+(visni*dvdxn-
     &              vissi*dvdxs)/sns(j)*vol
               dwdxt = (ww(i,j,k+1)-ww(i-1,j,k+1))/dxpw(i)
               dwdxb = (ww(i,j,k)-ww(i-1,j,k))/dxpw(i)
               visti = ctt(k)*vist + cbt(k)*visb
               visbi = cbb(k)*visb - cbt(k)*vist
               su(i,j,k) = su(i,j,k)+(visti*dwdxt-visbi*dwdxb)/
     &              stb(k)*vol
c     transient
               su(i,j,k) = su(i,j,k) + old_DEN(i,j,k)*vol*
     &                     (old_UU(i,j,k)-uu(i,j,k))/deltat
c     gravity
               su(i,j,k) = su(i,j,k) + (pt5*(den(i,j,k)+den(i-1,j,k))-
     &                                  den_ref)*grav*vol 
 100        continue
 110     continue
 120  continue
      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.2  2000/07/11 15:46:30  rawat
c added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
c
c Revision 1.1  2000/07/09 00:23:06  bbanerje
c Dummy version of uvelsrc copied from ucoef.
c
c
c*********************************************************************
