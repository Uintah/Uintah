c*********************************************************************
c
c $Id$
c
c*********************************************************************

      SUBROUTINE init(indexLow, indexHigh, NNN, NDH, 
     &     UU, VV, WW, PP, PP1,
     &     DEN, VIS,
     &     TKE, TED,
     &     FMU, DU, DV, DDW,
     &     SU, SP, FF,
     &     AUP, AVP, AWP,
     &     UP, VP, WP, FP, FPP, PT,
     &     VISP, DENP, VISCOS,
     &     RESU, RESV, RESW, RESP, RESF, RESTK, RESTE,
     &     RESUP, RESVP, RESWP, RESFP, RESPT,
     &     UCOR, VCOR, WCOR, PRCOR, FCOR, TKECOR, TEDCOR,
     &     UPCOR, VPCOR, WPCOR, FPCOR, PTCOR,
     &     spm, spu, spv, spw,
     &     INCELL,
     &     WORK,R0,SCFI,SCFD,
     &     DELTA,eps,
     &     trnArray,
     &     ierr)
C-----------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     INITIALIZATION ROUTINE     
C-----------------------------------------------------------------------

      implicit none

#include "param4.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer NNN, NDH
      double precision UU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision PP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision PP1(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision DEN(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision VIS(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision TKE(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision TED(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision FMU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision DU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision DV(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision DDW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision SU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision SP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision FF(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision AUP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision AVP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision AWP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision UP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision VP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision WP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision FP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision FPP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision PT(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision VISP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision DENP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision VISCOS
      double precision RESU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESV(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESF(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESTK(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESTE(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESUP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESVP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESWP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESFP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision RESPT(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision UCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision VCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision WCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision PRCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision FCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision TKECOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision TEDCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision UPCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision VPCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision WPCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision FPCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision PTCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision spm(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision spu(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision spv(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      double precision spw(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3)) 
      integer INCELL(6, indexLow(1):indexHigh(1), 
     &     indexLow(1):indexHigh(1))
      double precision WORK(NNN,NDH),R0(*),SCFI(*),SCFD(*)
      double precision DELTA, eps
      double precision trnArray(14)
      integer ierr

c*********************************************************************
c     Locals :
c*********************************************************************
      integer iBegGhost, iEndGhost, iBegActual, iEndActual
      integer jBegGhost, jEndGhost, jBegActual, jEndActual
      integer kBegGhost, kEndGhost, kBegActual, kEndActual

      double precision TRNCU,TRNCV,TRNCW,TRNCP,TRNCM,
     &     TRNCF,TRNCTK,TRNCTE,
     &     TRNCUP,TRNCVP,TRNCWP,TRNCFP,TRNCMP,TRNCPT

      integer i, j, k, minIJK, maxIJK, i1, i2, is

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      iBegGhost = indexLow(1)
      jBegGhost = indexLow(2)
      kBegGhost = indexLow(3)
      iBegActual = iBegGhost + 1
      jBegActual = jBegGhost + 1
      kBegActual = kBegGhost + 1
      iEndGhost = indexHigh(1)
      jEndGhost = indexHigh(2)
      kEndGhost = indexHigh(3)
      iEndActual = iEndGhost - 1
      jEndActual = jEndGhost - 1
      kEndActual = kEndGhost - 1

c*********************************************************************
c     Get the other variables
c*********************************************************************
      TRNCU = trnArray(1)
      TRNCV = trnArray(2)
      TRNCW = trnArray(3)
      TRNCP = trnArray(4)
      TRNCM = trnArray(5)
      TRNCF = trnArray(6)
      TRNCTK = trnArray(7)
      TRNCTE = trnArray(8)
      TRNCUP = trnArray(9)
      TRNCVP = trnArray(10)
      TRNCWP = trnArray(11)
      TRNCFP = trnArray(12)
      TRNCMP = trnArray(13)
      TRNCPT = trnArray(14)

C-----------------------------------------------------------------------
C     INITIALIZE ALL CARDINAL VARIABLES  
C-----------------------------------------------------------------------
      TRNCU  = ZERO
      TRNCV  = ZERO
      TRNCW  = ZERO
      TRNCP  = ZERO
      TRNCM  = ZERO
      TRNCTK = ZERO
      TRNCTE = ZERO
      TRNCF  = ZERO
      TRNCUP = ZERO
      TRNCVP = ZERO
      TRNCWP = ZERO
      TRNCFP = ZERO
      TRNCMP = ZERO
      TRNCPT = ZERO

      DO 300 K = kBegGhost,kEndGhost
         DO 200 J = jBegGhost,jEndGhost
            DO 100 I = iBegGhost,iEndGhost
               UU(I,J,K) = ZERO
               VV(I,J,K) = ZERO
               WW(I,J,K) = ZERO
               PP(I,J,K) = ZERO
               PP1(I,J,K) = ZERO
               DEN(I,J,K) = ZERO
               VIS(I,J,K) = VISCOS
               TKE(I,J,K) = ZERO
               TED(I,J,K) = ZERO
               FMU(I,J,K) = ONE
               DU(I,J,K) = ZERO
               DV(I,J,K) = ZERO
               DDW(I,J,K) = ZERO
               SU(I,J,K) = ZERO
               SP(I,J,K) = ZERO
               FF(I,J,K) = ZERO
               AUP(I,J,K) = ONE
               AVP(I,J,K) = ONE
               AWP(I,J,K) = ONE
               UP(I,J,K) = ZERO
               VP(I,J,K) = ZERO
               WP(I,J,K) = ZERO
               FP(I,J,K) = ZERO
               FPP(I,J,K) = ZERO
               PT(I,J,K) = ZERO
               VISP(I,J,K) = ZERO
               DENP(I,J,K) = ZERO
               RESU(I,J,K) = ZERO
               RESV(I,J,K) = ZERO
               RESW(I,J,K) = ZERO
               RESP(I,J,K) = ZERO
               RESF(I,J,K) = ZERO
               RESTK(I,J,K) = ZERO
               RESTE(I,J,K) = ZERO
               RESUP(I,J,K) = ZERO
               RESVP(I,J,K) = ZERO
               RESWP(I,J,K) = ZERO
               RESFP(I,J,K) = ZERO
               RESPT(I,J,K) = ZERO
               UCOR(I,J,K) = ZERO
               VCOR(I,J,K) = ZERO
               WCOR(I,J,K) = ZERO
               PRCOR(I,J,K) = ZERO
               FCOR(I,J,K) = ZERO
               TKECOR(I,J,K) = ZERO
               TEDCOR(I,J,K) = ZERO
               UPCOR(I,J,K) = ZERO
               VPCOR(I,J,K) = ZERO
               WPCOR(I,J,K) = ZERO
               FPCOR(I,J,K) = ZERO
               PTCOR(I,J,K) = ZERO
               spm(i,j,k) = zero
               spu(i,j,k) = zero
               spv(i,j,k) = zero
               spw(i,j,k) = zero
 100        CONTINUE
 200     CONTINUE
 300  CONTINUE
      minIJK = min(iBegGhost, jBegGhost, kBegGhost)
      maxIJK = max(iEndGhost, jEndGhost, kEndGhost)
      DO 420 IS=1,6
         DO 410 I1=minIJK, maxIJK
            DO 400 I2=minIJK, maxIJK
               INCELL(IS,I1,I2) = 0
               IF (IS.EQ.2) INCELL(IS,I1,I2) = iEndGhost + 1
               IF (IS.EQ.4) INCELL(IS,I1,I2) = jEndGhost + 1
               IF (IS.EQ.6) INCELL(IS,I1,I2) = kEndGhost + 1
 400        CONTINUE
 410     CONTINUE
 420  CONTINUE

      DO 700 I = 1,NNN
         SCFI(I) = ONE
         SCFD(I) = ONE
         R0(I) = ZERO
 700  CONTINUE
      DELTA = sqrt(eps)
      DO 900 J=1,NDH
         DO 800 I=1,NNN
            WORK(I,J) = ZERO
 800     CONTINUE
 900  CONTINUE

c*********************************************************************
c     Set the other variables
c*********************************************************************
      trnArray(1) = TRNCU
      trnArray(2) = TRNCV
      trnArray(3) = TRNCW
      trnArray(4) = TRNCP
      trnArray(5) = TRNCM
      trnArray(6) = TRNCF
      trnArray(7) = TRNCTK
      trnArray(8) = TRNCTE
      trnArray(9) = TRNCUP
      trnArray(10) = TRNCVP
      trnArray(11) = TRNCWP
      trnArray(12) = TRNCFP
      trnArray(13) = TRNCMP
      trnArray(14) = TRNCPT

      RETURN
      END

c*********************************************************************
c
c $Log$
c Revision 1.1  2000/04/12 20:31:47  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
