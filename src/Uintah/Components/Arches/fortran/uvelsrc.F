c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE uvelsrc(domLoU, domHiU, idxLoU, idxHiU, UU,
     &     CESAV, CWSAV, CNSAV, CSSAV, CTSAV, CBSAV,
     &     AP, AE, AW, AN, AS, AT, AB,
     &     DU,
     &     domLoV, domHiV, VV,
     &     domLoW, domHiW, WW,
     &     domLo, domHi, DEN, VIS, 
     &     DELTAT,
     &     CEEU, CWEU, CWWU, CNN, CSN, CSS,
     &     CTT, CBT, CBB, SEWU, SNS, STB,
     &     DXEPU, DXPWU, DYNP, DYPS, DZTP, DZPB, 
     &     FAC1U, FAC2U, FAC3U, FAC4U, 
     &     IESDU, IWSDU,
     &     ENFAC, SFAC, TFAC, BFAC,
     &     ierr)
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

      implicit none

c #include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer domLoU(3), domHiU(3), idxLoU(3), idxHiU(3)
      double precision UU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AP(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AE(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AW(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AN(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AS(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AT(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision AB(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CESAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CWSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CNSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CSSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CTSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision CBSAV(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
     &     domLoU(3):domHiU(3))
      double precision DU(domLoU(1):domHiU(1), domLoU(2):domHiU(2), 
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
      double precision VIS(domLo(1):domHi(1), domLo(2):domHi(2), 
     &     domLo(3):domHi(3))
      double precision DELTAT
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
      double precision SNS(domLo(2):domHi(2))
      double precision STB(domLo(3):domHi(3))
      double precision DXEPU(domLo(1):domHi(1))
      double precision DXPWU(domLo(1):domHi(1))
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

      logical L2UP, LSMPT, LCEND

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

c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      DO 220 K = KSTU,KENDU
         DO 210 J = JSTU,JENDU
            DO 200 I = ISTU,IENDU   
C-----------------------------------------------------------------------
C     ASSEMBLE SOURCE COEFFICIENTS
C-----------------------------------------------------------------------
C            IF (LTEST) SU(I,J,K) = SU(I,J,K) + USU(I,J,K)*VOL
            SP(I,J,K) = 0.0
            IF (LTIM) THEN
              APO(I,J,K) = DENO(I,J,K)*VOL/DTIME
              SU(I,J,K) = SU(I,J,K) + APO(I,J,K)*(UO(I,J,K)-U(I,J,K))
            END IF
            DUDXE  = (U(I+1,J,K)-U(I,J,K))/SEW(I)
            DUDXW  = (U(I,J,K)-U(I-1,J,K))/SEW(I-1)
            VISEI = CEEU(I)*VISE + CWEU(I)*VISW
            VISWI = CWWU(I)*VISW - CWEU(I)*VISE
            SU(I,J,K) = SU(I,J,K)+(VISEI*DUDXE-VISWI*DUDXW)/
     &                  SEWU(I)*VOL
            DVDXN  = (V(I,J+1,K)-V(I-1,J+1,K))/DXPW(I)
            DVDXS  = (V(I,J,K)-V(I-1,J,K))/DXPW(I)
            VISNI = CNN(J)*VISN + CSN(J)*VISS
            VISSI = CSS(J)*VISS - CSN(J)*VISN
            SU(I,J,K) = SU(I,J,K)+(VISNI*RV(J+1)*DVDXN-VISSI*RV(J)*
     &                  DVDXS)/(R(J)*SNS(J))*VOL
            DWDXT = (W(I,J,K+1)-W(I-1,J,K+1))/DXPW(I)
            DWDXB = (W(I,J,K)-W(I-1,J,K))/DXPW(I)
            VISTI = CTT(K)*VIST + CBT(K)*VISB
            VISBI = CBB(K)*VISB - CBT(K)*VIST
            SU(I,J,K) = SU(I,J,K)+(VISTI*DWDXT-VISBI*DWDXB)/
     &                  (R(J)*STB(K))*VOL
            SU(I,J,K) = SU(I,J,K)+0.5*(DEN(I,J,K)+DEN(I-1,J,K))*
     &                  GX*VOL
     &                  -DEN(IPREF,JPREF,KPREF)*GX*VOL
            IF (.NOT.LTEST.AND.LKETM.AND..NOT.LMLTM)
     &           SU(I,J,K) = SU(I,J,K) + 0.6667*(VOL/DXPW(I))*
     &           (DEN(I-1,J,K)*TKE(I-1,J,K)-DEN(I,J,K)*TKE(I,J,K))
            SU(I,J,K) = SU(I,J,K) + SPU(I,J,K)
            SUL(I,J,K) = SU(I,J,K)
  200     CONTINUE
C-----------------------------------------------------------------------
C    THE NONLINEAR K-EPSILON MODEL OPTION
C-----------------------------------------------------------------------
         IF (LNLKM) THEN
          DO 204 I = ISTART,NIM1
C---------------------------------
C    COMPUTE VOLUME
C---------------------------------
            VOL = SEWU(I)*SNS(J)*RA(J)*STB(K)
C-----------------------------------------------------------------------
C    COEFFICIENTS
C-----------------------------------------------------------------------
            IF (PCELL(I,J,K).NE.WALL) THEN
              COFEO = VIS(I,J,K)*VIS(I,J,K)/(DEN(I,J,K)
     &               *TKE(I,J,K))
            ELSE
              COFEO = 0.0
            END IF
            IF (PCELL(I-1,J,K).NE.WALL) THEN
              COFWO = VIS(I-1,J,K)*VIS(I-1,J,K)/(DEN(I-1,J,K)
     &               *TKE(I-1,J,K))
            ELSE
              COFWO = 0.0
            END IF
C........
            COFE = CEEU(I)*COFEO + CWEU(I)*COFWO
            COFW = CWWU(I)*COFWO - CWEU(I)*COFEO
C........
            IF (PCELL(I,J+1,K).NE.WALL) THEN
              COFNE = VIS(I,J+1,K)*VIS(I,J+1,K)/(DEN(I,J+1,K)
     &               *TKE(I,J+1,K))
            ELSE
              COFNE = COFEO
            END IF
            IF (PCELL(I-1,J+1,K).NE.WALL) THEN
              COFNW = VIS(I-1,J+1,K)*VIS(I-1,J+1,K)
     &               /(DEN(I-1,J+1,K)*TKE(I-1,J+1,K))
            ELSE
              COFNW = COFWO
            END IF
C.............................................................
          IF (COFEO.NE.0.0.AND.COFWO.NE.0.0.AND.COFNE.NE.0.0
     &        .AND.COFNW.NE.0.0) THEN
              COFNO = 4.0/((1.0/COFEO)+(1.0/COFWO)+(1.0/COFNE)
     &                   +(1.0/COFNW))
          ELSE
              COFNO = 0.25*(COFEO + COFWO + COFNE + COFNW)
          END IF
C.............................................................
            IF (PCELL(I,J-1,K).NE.WALL) THEN
              COFSE = VIS(I,J-1,K)*VIS(I,J-1,K)/(DEN(I,J-1,K)
     &               *TKE(I,J-1,K))
            ELSE
              COFSE = COFEO
            END IF
            IF (PCELL(I-1,J-1,K).NE.WALL) THEN
              COFSW = VIS(I-1,J-1,K)*VIS(I-1,J-1,K)
     &               /(DEN(I-1,J-1,K)*TKE(I-1,J-1,K))
            ELSE
              COFSW = COFWO
            END IF
C.............................................................
          IF (COFEO.NE.0.0.AND.COFWO.NE.0.0.AND.COFSE.NE.0.0
     &        .AND.COFSW.NE.0.0) THEN
              COFSO = 4.0/((1.0/COFEO)+(1.0/COFWO)+(1.0/COFSE)
     &                   +(1.0/COFSW))
          ELSE
              COFSO = 0.25*(COFEO + COFWO + COFSE + COFSW)
          END IF
C.............................................................
              COFN = CNN(J)*COFNO + CSN(J)*COFSO
              COFS = CSS(J)*COFSO - CSN(J)*COFNO
C........
            IF (PCELL(I,J,K+1).NE.WALL) THEN
              COFTE = VIS(I,J,K+1)*VIS(I,J,K+1)/(DEN(I,J,K+1)
     &                *TKE(I,J,K+1))
            ELSE 
              COFTE = COFEO
            END IF
            IF (PCELL(I-1,J,K+1).NE.WALL) THEN
              COFTW = VIS(I-1,J,K+1)*VIS(I-1,J,K+1)
     &               /(DEN(I-1,J,K+1)*TKE(I-1,J,K+1))
            ELSE
              COFTW = COFWO
            END IF
C.............................................................
          IF (COFEO.NE.0.0.AND.COFWO.NE.0.0.AND.COFTE.NE.0.0
     &        .AND.COFTW.NE.0.0) THEN
              COFTO = 4.0/((1.0/COFEO)+(1.0/COFWO)+(1.0/COFTE)
     &                   +(1.0/COFTW))
          ELSE
              COFTO = (COFEO + COFWO + COFTE + COFTW)
          END IF
C.............................................................
            IF (PCELL(I,J,K-1).NE.WALL) THEN
              COFBE = VIS(I,J,K-1)*VIS(I,J,K-1)/(DEN(I,J,K-1)
     &                *TKE(I,J,K-1))
            ELSE
              COFBE = COFEO
            END IF
            IF (PCELL(I-1,J,K-1).NE.WALL) THEN
              COFBW = VIS(I-1,J,K-1)*VIS(I-1,J,K-1)
     &               /(DEN(I-1,J,K-1)*TKE(I-1,J,K-1))
            ELSE
              COFBW = COFWO
            END IF
C............................................................
          IF (COFEO.NE.0.0.AND.COFWO.NE.0.0.AND.COFBE.NE.0.0
     &        .AND.COFBW.NE.0.0) THEN 
              COFBO = 4.0/((1.0/COFEO)+(1.0/COFWO)+(1.0/COFBE)
     &                 +(1.0/COFBW))
          ELSE
              COFBO = 0.25*(COFEO + COFWO + COFBE + COFBW)
          END IF
C............................................................
            COFT = CTT(K)*COFTO + CBT(K)*COFBO
            COFB = CBB(K)*COFBO - CBT(K)*COFTO
C-----------------------------------------------------------------------
C          d/dx[c(du/dx)**2]
C-----------------------------------------------------------------------
c              DUDXE = (U(I+1,J,K)-U(I,J,K))/SEW(I)
c              DUDXW = (U(I,J,K)-U(I-1,J,K))/SEW(I-1)
C-------------
c          SU(I,J,K) = SU(I,J,K) - 4.482*VOL*(COFE*DUDXE*DUDXE
c     &                 - COFW*DUDXW*DUDXW)/DXPW(I)
C-----------------------------------------------------------------------
C          d/dx[c(dv/dy)**2]
C-----------------------------------------------------------------------
c              DVDYE = (V(I,J+1,K)-V(I,J,K))/SNS(J)
c              DVDYW = (V(I-1,J+1,K)-V(I-1,J,K))/SNS(J)
C-------------
c           SU(I,J,K) = SU(I,J,K) - 6.72*VOL*(COFE*DVDYE*DVDYE
c     &                 - COFW*DVDYW*DVDYW)/DXPW(I)
C-----------------------------------------------------------------------
C          d/dx[c(dw/dz)**2]
C-----------------------------------------------------------------------
c              DWDZE = (W(I,J,K+1)-W(I,J,K))/(STB(K)*RA(J))
c              DWDZW = (W(I-1,J,K+1)-W(I-1,J,K))/(STB(K)*RA(J))
C-------------
c           SU(I,J,K) = SU(I,J,K) - 6.72*VOL*(COFE*DWDZE*DWDZE
c     &                 - COFW*DWDZW*DWDZW)/DXPW(I)     
C-----------------------------------------------------------------------
C          d/dx[C(dw/dx)**2]
C-----------------------------------------------------------------------
              WE = TFAC(K)*0.5*(W(I+1,J,K+1)+W(I,J,K+1))
     &            +BFAC(K)*0.5*(W(I+1,J,K)+W(I,J,K))
              WP = TFAC(K)*0.5*(W(I-1,J,K+1)+W(I,J,K+1))
     &            +BFAC(K)*0.5*(W(I-1,J,K)+W(I,J,K))
              WN = TFAC(K)*0.5*(W(I-1,J+1,K+1)+W(I,J+1,K+1))
     &            +BFAC(K)*0.5*(W(I-1,J+1,K)+W(I,J+1,K)) 
              WS = TFAC(K)*0.5*(W(I-1,J-1,K+1)+W(I,J-1,K+1))
     &            +BFAC(K)*0.5*(W(I-1,J-1,K)+W(I,J-1,K))
            IF (I.GT.2) THEN
              WW = TFAC(K)*0.5*(W(I-2,J,K+1)+W(I-1,J,K+1))
     &            +BFAC(K)*0.5*(W(I-2,J,K)+W(I-1,J,K))
            ELSE
              WW = WP
            END IF
C-------------
              DWDXE = (WE-WP)/SEW(I)
              DWDXW = (WP-WW)/SEW(I-1)
C-------------
            SU(I,J,K) = SU(I,J,K)-1.68*VOL*((COFE*DWDXE*DWDXE-
     &                  COFW*DWDXW*DWDXW)/DXPW(I))
C-----------------------------------------------------------------------
C          d/dx[C(dw/dy)**2]
C-----------------------------------------------------------------------
              WNP = TFAC(K)*0.5*(W(I,J,K+1)+W(I,J+1,K+1))
     &             +BFAC(K)*0.5*(W(I,J,K)+W(I,J+1,K))
              WSP = TFAC(K)*0.5*(W(I,J-1,K+1)+W(I,J,K+1))
     &             +BFAC(K)*0.5*(W(I,J-1,K)+W(I,J,K))
              WNWP = TFAC(K)*0.5*(W(I-1,J,K+1)+W(I-1,J+1,K+1))
     &              +BFAC(K)*0.5*(W(I-1,J,K)+W(I-1,J+1,K))
              WSWP = TFAC(K)*0.5*(W(I-1,J-1,K+1)+W(I-1,J,K+1))
     &              +BFAC(K)*0.5*(W(I-1,J-1,K)+W(I-1,J,K))
C-------------
              DWDYE = (WNP-WSP)/SNS(J)
              DWDYW = (WNWP-WSWP)/SNS(J)
C-------------
            SU(I,J,K) = SU(I,J,K)-1.68*VOL*2.0*(COFE*DWDYE*DWDYE-
     &                  COFW*DWDYW*DWDYW)/DXPW(I)
C-----------------------------------------------------------------------
C         d/dx[C(dv/dx)**2)]
C-----------------------------------------------------------------------
              VE = ENFAC(J)*0.5*(V(I+1,J+1,K)+V(I,J+1,K))
     &             +SFAC(J)*0.5*(V(I+1,J,K)+V(I,J,K))
              VP = ENFAC(J)*0.5*(V(I,J+1,K)+V(I-1,J+1,K))
     &             +SFAC(J)*0.5*(V(I,J,K)+V(I-1,J,K))
            IF (I.GT.2) THEN
              VW = ENFAC(J)*0.5*(V(I-1,J+1,K)+V(I-2,J+1,K))
     &             +SFAC(J)*0.5*(V(I-1,J,K)+V(I-2,J,K))
            ELSE
              VW = VP
            END IF
C-------------
              DVDXE = (VE-VP)/SEW(I)
              DVDXW = (VP-VW)/SEW(I-1)
C-------------
            SU(I,J,K) = SU(I,J,K)-1.68*VOL*(COFE*DVDXE*DVDXE-
     &                  COFW*DVDXW*DVDXW)/DXPW(I)
C-----------------------------------------------------------------------
C         d/dx[C(dv/dz)**2]
C-----------------------------------------------------------------------
              VTP = ENFAC(J)*0.5*(V(I,J+1,K+1)+V(I,J+1,K))
     &             +SFAC(J)*0.5*(V(I,J,K+1)+V(I,J,K))
              VBP = ENFAC(J)*0.5*(V(I,J+1,K)+V(I,J+1,K-1))
     &             +SFAC(J)*0.5*(V(I,J,K)+V(I,J,K-1))
              VTW = ENFAC(J)*0.5*(V(I-1,J+1,K+1)+V(I-1,J+1,K))
     &              +SFAC(J)*0.5*(V(I-1,J,K+1)+V(I-1,J,K))
              VBW = ENFAC(J)*0.5*(V(I-1,J+1,K)+V(I-1,J+1,K-1))
     &              +SFAC(J)*0.5*(V(I-1,J,K)+V(I-1,J,K-1))
C-------------
              DVDZE = (VTP-VBP)/(STB(K)*RA(J))
              DVDZW = (VTW-VBW)/(STB(K)*RA(J))
C-------------
            SU(I,J,K) = SU(I,J,K)-1.68*2.0*VOL*(COFE*DVDZE*DVDZE-
     &                  COFW*DVDZW*DVDZW)/DXPW(I)
C-----------------------------------------------------------------------
C ***        d/dx[c(du/dz)**2]
C-----------------------------------------------------------------------
c              UT =  EFAC(I)*0.5*(U(I+1,J,K+1)+U(I+1,J,K))
c     &             +WFAC(I)*0.5*(U(I,J,K+1)+U(I,J,K))
c              UB =  EFAC(I)*0.5*(U(I+1,J,K)+U(I+1,J,K-1))
c     &             +WFAC(I)*0.5*(U(I,J,K)+U(I,J,K-1))
c              UTW = EFAC(I)*0.5*(U(I,J,K+1)+U(I,J,K))
c     &             +WFAC(I)*0.5*(U(I-1,J,K+1)+U(I-1,J,K))
c              UBW = EFAC(I)*0.5*(U(I,J,K)+U(I,J,K-1))
c     &             +WFAC(I)*0.5*(U(I-1,J,K)+U(I-1,J,K-1))
C-------------
c              DUDZE = (UT-UB)/(STB(K)*RA(J))
c              DUDZW = (UTW-UBW)/(STB(K)*RA(J))
C-------------
c            SU(I,J,K) = SU(I,J,K)-1.68*2.33*VOL*(COFE*DUDZE*DUDZE-
c     &                  COFW*DUDZW*DUDZW)/DXPW(I)
C-----------------------------------------------------------------------
C ***        d/dx[C(du/dy)**2]
C-----------------------------------------------------------------------
c              UNE = EFAC(I)*0.5*(U(I+1,J+1,K)+U(I+1,J,K))
c     &             +WFAC(I)*0.5*(U(I,J+1,K)+U(I,J,K))
c              USE = EFAC(I)*0.5*(U(I+1,J-1,K)+U(I+1,J,K))
c     &             +WFAC(I)*0.5*(U(I,J-1,K)+U(I,J,K))
c              UNW = EFAC(I)*0.5*(U(I,J+1,K)+U(I,J,K))
c     &             +WFAC(I)*0.5*(U(I-1,J+1,K)+U(I-1,J,K))
c              USW = EFAC(I)*0.5*(U(I,J-1,K)+U(I,J,K))
c     &             +WFAC(I)*0.5*(U(I-1,J-1,K)+U(I-1,J,K))
C-------------
c              DUDYE = (UNE-USE)/SNS(J)
c              DUDYW = (UNW-USW)/SNS(J)
C-------------
c            SU(I,J,K) = SU(I,J,K)-1.68*2.33*VOL*(COFE*DUDYE*DUDYE-
c     &                  COFW*DUDYW*DUDYW)/DXPW(I)
C-----------------------------------------------------------------------
C         d/dy[C(dw/dx)(dw/dy)]
C-----------------------------------------------------------------------
              DWDXN = (WNP-WNWP)/DXPW(I)
              DWDXS = (WSP-WSWP)/DXPW(I)
              DWDYN = (WN-WP)/DYNP(J)
              DWDYS = (WP-WS)/DYPS(J)
            SU(I,J,K) = SU(I,J,K) + 1.68*VOL*(COFN*DWDYN*DWDXN -
     &                  COFS*DWDYS*DWDXS)/SNS(J)
C-----------------------------------------------------------------------
C         d/dz[C(dv/dx)(dv/dz)]
C-----------------------------------------------------------------------
              VT = ENFAC(J)*0.5*(V(I,J+1,K+1)+V(I-1,J+1,K+1))
     &            +SFAC(J)*0.5*(V(I,J,K+1)+V(I-1,J,K+1))
              VB = ENFAC(J)*0.5*(V(I,J+1,K-1)+V(I-1,J+1,K-1))
     &            +SFAC(J)*0.5*(V(I,J,K-1)+V(I-1,J,K-1))
C-------------
              DVDXT = (VTP-VTW)/DXPW(I)
              DVDXB = (VBP-VBW)/DXPW(I)
              DVDZT = (VT-VP)/DZTP(K)
              DVDZB = (VP-VB)/DZPB(K)
            SU(I,J,K) = SU(I,J,K)+1.68*VOL*(COFT*DVDXT*DVDZT-
     &                  COFB*DVDXB*DVDZB)/(STB(K)*RA(J))
C------------------------------------------------------------------
  204     CONTINUE
         END IF
C------------------------------------------------------------------ 
          DO 205 I = 2,NIM1
            IF (PCELL(I-1,J,K).EQ.WALL.OR.PCELL(I,J,K).EQ.WALL) THEN
              SU(I,J,K) = GREAT*0.0
              SP(I,J,K) = -GREAT
            END IF
  205     CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/07/09 00:23:06  bbanerje
c Dummy version of uvelsrc copied from ucoef.
c
c
c*********************************************************************
