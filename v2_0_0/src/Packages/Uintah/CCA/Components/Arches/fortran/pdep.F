c*********************************************************************
c
c
c*********************************************************************

#if 1
#include <Packages/Uintah/CCA/Components/Arches/fortran/pdep_fort.h>
#else
      SUBROUTINE pdep(indexLow, indexHigh, NNN, NI, NJ, NK,
     &     NITER, PCELL, 
     $     UU,VV,WW,PP,
     &     DEN,DENP,PDEN,
     &     RWL1,RWL2,RWINF,
     $     RESRP0,TRNCP0,
     &     RESP,PRCOR,
     &     SCF,
     &     PFIX,
     &     RCV,
     $     CPL,
     &     GAMP,URFP,NSWPP,
     &     TSOLV,THETA,
     &     SEWU,SNSV,STBW,
     $     AUE,AUW,AUN,AUS,AUT,AUB,AUP,SUU,
     $     AVE,AVW,AVN,AVS,AVT,AVB,AVP,SVU,
     $     AWE,AWW,AWN,AWS,AWT,AWB,AWP,SWU,
     $     AE,AW,AN,AS,AT,AB,AP,SU,SP,
     &     SIGN,
     &     SPM,
     $     SEW,SNS,STB,DXEP,DXPW,DYNP,DYPS,DZTP,DZPB,
     &     RR,RV,RA,
     &     IEQN,ISPP,IMODE,ILCNT,ISOPT,IPREF,JPREF,KPREF,
     $     IINF,JINF,KINF,IEQNIN,NTOTAL,
     &     INCELL,
     &     IJ,IK,NIJK,
     $     LP,LPARL,LCSCF,LLEFT,LCALP,LTEST,
     $     LWRAP,LSYMB,LPCON,LEXCP,
     &     LPRDAV, LPRDJK, LPRDKK,
     &     tmpXYZArr1, tmpXYZArr2,
     &     tmpXArr1, tmpXArr2,
     $     NEQ, CLABEL,
     &     ierr)
*----------------------------------------------------------------------- 
*     Written by Seshadri Kumar, August 4, 1995
*     Last modified, September 16, 1998
*
*     Passed variables (partial list):
*
*	ILCNT	Total number of calls to subroutine linegs in the
*		entire calculation (i/p and o/p)
*-----------------------------------------------------------------------
      implicit none

#include "param4.h"
#include "pcell.h"

c*********************************************************************
c     Arguments :
c*********************************************************************
      integer indexLow(3)
      integer indexHigh(3)
      integer NNN, NI, NJ, NK
      integer NITER
      integer PCELL(indexLow(1):indexHigh(1), indexLow(2):indexHigh(2), 
     &     indexLow(3):indexHigh(3))
      double precision UU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision VV(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision WW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision PP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision DEN(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision DENP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision PDEN
      double precision RWL1, RWL2, RWINF
      double precision RESRP0, TRNCP0
      double precision RESP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision PRCOR(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision SCF(NNN)
      double precision PFIX
      double precision RCV(indexLow(2):indexHigh(2))
      double precision CPL(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision GAMP, URFP
      integer NSWPP
      double precision TSOLV, THETA
      double precision SEWU(indexLow(1):indexHigh(1))
      double precision SNSV(indexLow(2):indexHigh(2))
      double precision STBW(indexLow(3):indexHigh(3))
      double precision AUE(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AUW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AUN(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AUS(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AUT(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AUB(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AUP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision SUU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AVE(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AVW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AVN(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AVS(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AVT(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AVB(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AVP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision SVU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AWE(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AWW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AWN(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AWS(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AWT(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AWB(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AWP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision SWU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AE(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AW(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AN(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AS(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AT(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AB(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision AP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision SU(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision SP(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision SIGN
      double precision SPM(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2),indexLow(3):indexHigh(3))
      double precision SEW(indexLow(1):indexHigh(1))
      double precision SNS(indexLow(2):indexHigh(2))
      double precision STB(indexLow(3):indexHigh(3))
      double precision DXEP(indexLow(1):indexHigh(1))
      double precision DXPW(indexLow(1):indexHigh(1))
      double precision DYNP(indexLow(2):indexHigh(2))
      double precision DYPS(indexLow(2):indexHigh(2))
      double precision DZTP(indexLow(3):indexHigh(3))
      double precision DZPB(indexLow(3):indexHigh(3))
      double precision RR(indexLow(2):indexHigh(2))
      double precision RV(indexLow(2):indexHigh(2))
      double precision RA(indexLow(2):indexHigh(2))
      integer IEQN,ISPP,IMODE,ILCNT,ISOPT,IPREF,JPREF,KPREF
      integer IINF,JINF,KINF,IEQNIN,NTOTAL
      integer INCELL(6,indexLow(1):indexHigh(1),
     &     indexLow(1):indexHigh(1))
      integer IJ(indexLow(2):indexHigh(2))
      integer IK(indexLow(3):indexHigh(3))
      integer NIJK
      logical LP,LPARL,LCSCF,LLEFT,LCALP,LTEST, LWRAP,LSYMB,LPCON,LEXCP
      logical LPRDKK, LPRDAV, LPRDJK
      double precision tmpXYZArr1(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision tmpXYZArr2(indexLow(1):indexHigh(1),
     &     indexLow(2):indexHigh(2), indexLow(3):indexHigh(3))
      double precision tmpXArr1(indexLow(1):indexHigh(1))
      double precision tmpXArr2(indexLow(1):indexHigh(1))
      integer NEQ
      CHARACTER*4 CLABEL(NEQ+2)
      integer ierr

c*********************************************************************
c     Locals :
c*********************************************************************
      double precision 
     $     CONVS,CONVL,CONST,AREAN,AREAS,AREATB,AREAEW,EPSG,VOL,
     $     DENE,DENW,DENN,DENS,DENT,DENB,UE,UW,VN,VS,WT,WB,SMP,
     $     R_NORM,RESORP,TRNCP,PPREF,CE,CW,CN,CS,CT,CB
      integer i,j,k
      integer IST, JST, KST, IEND, JEND, KEND
      integer NJM1
      LOGICAL LPAR

c*********************************************************************
c     Start :
c*********************************************************************
      IST = indexLow(1)+1
      JST = indexLow(2)+1
      KST = indexLow(3)+1
      IEND = indexHigh(1)-1
      JEND = indexHigh(2)-1
      KEND = indexHigh(3)-1
      NJM1 = JEND
  
C-----------------------------------------------------------------------
      CONVS = ZERO
      CONST = ONE
      IF (LPARL.AND..NOT.LEXCP) THEN
         LPAR = .TRUE.
      ELSE
         LPAR = .FALSE.
      ENDIF
C-----------------------------------------------------------------------
C     ASSEMBLY OF COEFFICIENTS  
C-----------------------------------------------------------------------
      DO 120 K = KST,KEND
         DO 110 J = JST,JEND
            DO 100 I = IST,IEND
C-----------------------------------------------------------------------
C     COMPUTE AREAS AND VOLUME
C-----------------------------------------------------------------------
               AREAN = RV(J+1)*SEW(I)*STB(K)
               AREAS = RV(J)*SEW(I)*STB(K)
               AREAEW = RA(J)*SNS(J)*STB(K)
               AREATB = SNS(J)*SEW(I)
               VOL = RA(J)*SNS(J)*SEW(I)*STB(K)
               EPSG = ONE
               IF (LCALP .AND. .NOT.LP) THEN
                  CALL CALEPS(indexLow, indexHigh, 
     &                 DENP,PDEN,I,J,K,EPSG)
               ENDIF
C-----------------------------------------------------------------------
C     CALCULATE COEFFICIENTS
C-----------------------------------------------------------------------
               DENE = PT5*(DEN(I,J,K)+DEN(I+1,J,K))
               DENW = PT5*(DEN(I,J,K)+DEN(I-1,J,K))
               DENN = PT5*(DEN(I,J,K)+DEN(I,J+1,K))
               DENS = PT5*(DEN(I,J,K)+DEN(I,J-1,K))
               DENT = PT5*(DEN(I,J,K)+DEN(I,J,K+1))
               DENB = PT5*(DEN(I,J,K)+DEN(I,J,K-1))
               AE(I,J,K) = DENE*AREAEW*AREAEW*SEWU(I+1)/
     &              (DXEP(I)*AUP(I+1,J,K))*EPSG
               AW(I,J,K) = DENW*AREAEW*AREAEW*SEWU(I)/
     &              (DXPW(I)*AUP(I,J,K))*EPSG
               IF (J.EQ.NJM1) THEN
                  AN(I,J,K) = DENN*AREAN*SEW(I)*SNSV(J+1)*RV(J+1)
     &                 *STB(K)/(DYNP(J)*AVP(I,J+1,K))*EPSG
               ELSE
                  AN(I,J,K) = DENN*AREAN*SEW(I)*SNSV(J+1)*PT5*
     &              (RCV(J+1)+RCV(J+2))*STB(K)/
     $                 (DYNP(J)*AVP(I,J+1,K))*EPSG
               ENDIF
               AS(I,J,K) = DENS*AREAS*SEW(I)*SNSV(J)*PT5*(RCV(J)+
     &              RCV(J+1))*STB(K)/(DYPS(J)*AVP(I,J,K))*EPSG
               AT(I,J,K) = DENT*AREATB*AREATB*STBW(K+1)/
     &              (DZTP(K)*AWP(I,J,K+1))*EPSG
               AB(I,J,K) = DENB*AREATB*AREATB*STBW(K)/
     &              (DZPB(K)*AWP(I,J,K)) *EPSG
C-----------------------------------------------------------------------
C     CALCULATE PSEUDOVELOCITIES FOR BOUNDARIES 
C     (setting uhat = u on boundaries makes it easy to set bcs. for
C     the pressure)
C-----------------------------------------------------------------------
               UE = UU(I+1,J,K)
               UW = UU(I,J,K)
               VN = VV(I,J+1,K)
               VS = VV(I,J,K)
               WT = WW(I,J,K+1)
               WB = WW(I,J,K)
C-----------------------------------------------------------------------
C     CALCULATE PSEUDOVELOCITIES WHEN NOT ON BOUNDARIES
C-----------------------------------------------------------------------
               IF (I.NE.2 .AND. PCELL(I-1,J,K).EQ.FFIELD)
     &              UW = (AUE(I,J,K)*UU(I+1,J,K)+
     $                    AUW(I,J,K)*UU(I-1,J,K)+
     &                    AUN(I,J,K)*UU(I,J+1,K)+
     $                    AUS(I,J,K)*UU(I,J-1,K)+
     &                    AUT(I,J,K)*UU(I,J,K+1)+
     $                    AUB(I,J,K)*UU(I,J,K-1)+
     &                    SUU(I,J,K))/AUP(I,J,K)
               IF (I.NE.IEND .AND. PCELL(I+1,J,K).EQ.FFIELD)
     &              UE = (AUE(I+1,J,K)*UU(I+2,J,K)+
     &                    AUW(I+1,J,K)*UU(I,J,K)+
     $                    AUN(I+1,J,K)*UU(I+1,J+1,K)+
     &                    AUS(I+1,J,K)*UU(I+1,J-1,K)+
     $                    AUT(I+1,J,K)*UU(I+1,J,K+1)+
     &                    AUB(I+1,J,K)*UU(I+1,J,K-1)+
     $                    SUU(I+1,J,K))/AUP(I+1,J,K)
               IF (J.NE.2 .AND. PCELL(I,J-1,K).EQ.FFIELD)
     &              VS = (AVE(I,J,K)*VV(I+1,J,K)+
     $                    AVW(I,J,K)*VV(I-1,J,K)+
     &                    AVN(I,J,K)*VV(I,J+1,K)+
     $                    AVS(I,J,K)*VV(I,J-1,K)+
     &                    AVT(I,J,K)*VV(I,J,K+1)+
     $                    AVB(I,J,K)*VV(I,J,K-1)+
     &                    SVU(I,J,K))/AVP(I,J,K)
               IF (J.NE.JEND .AND. PCELL(I,J+1,K).EQ.FFIELD)
     &              VN = (AVE(I,J+1,K)*VV(I+1,J+1,K)+
     &                    AVW(I,J+1,K)*VV(I-1,J+1,K)+
     $                    AVN(I,J+1,K)*VV(I,J+2,K)+
     &                    AVS(I,J+1,K)*VV(I,J,K)+
     $                    AVT(I,J+1,K)*VV(I,J+1,K+1)+
     &                    AVB(I,J+1,K)*VV(I,J+1,K-1)+
     $                    SVU(I,J+1,K))/AVP(I,J+1,K)
               IF (K.NE.2 .AND. PCELL(I,J,K-1).EQ.FFIELD)
     &              WB = (AWE(I,J,K)*WW(I+1,J,K)+
     $                    AWW(I,J,K)*WW(I-1,J,K)+
     &                    AWN(I,J,K)*WW(I,J+1,K)+
     $                    AWS(I,J,K)*WW(I,J-1,K)+
     &                    AWT(I,J,K)*WW(I,J,K+1)+
     $                    AWB(I,J,K)*WW(I,J,K-1)+
     &                    SWU(I,J,K))/AWP(I,J,K)
               IF (K.NE.KEND .AND. PCELL(I,J,K+1).EQ.FFIELD)
     &              WT = (AWE(I,J,K+1)*WW(I+1,J,K+1)+
     &                    AWW(I,J,K+1)*WW(I-1,J,K+1)+
     $                    AWN(I,J,K+1)*WW(I,J+1,K+1)+
     &                    AWS(I,J,K+1)*WW(I,J-1,K+1)+
     $                    AWT(I,J,K+1)*WW(I,J,K+2)+
     &                    AWB(I,J,K+1)*WW(I,J,K)+
     $                    SWU(I,J,K+1))/AWP(I,J,K+1)   

C-----CALCULATE SOURCE TERMS--------------------------------------------

               CE = DENE*AREAEW
               CW = DENW*AREAEW
               CN = DENN*AREAN
               CS = DENS*AREAS
               CT = DENT*AREATB
               CB = DENB*AREATB
               SMP = CN*VN-CS*VS+CE*UE-CW*UW+CT*WT-CB*WB-
     $              sign*spm(i,j,k)
               IF (PCELL(I,J,K).EQ.WALL) THEN
                  CALL FIXVAL(indexLow, indexHigh,
     &                 ZERO,AE,AW,AN,AS,AT,AB,SU,SP,
     &                 CONST,I,J,K)
               ELSE IF (LTEST) THEN
                  SP(I,J,K) = ZERO
                  SU(I,J,K) = -SMP + CPL(I,J,K)*VOL
               ELSE
                  SP(I,J,K) = ZERO
                  SU(I,J,K) = -SMP
                  IF (SMP.NE.ZERO)
     &                 CONVS = CONVS + ABS(SMP)/MAX(ABS(CN*VN),
     $                 ABS(CS*VS),ABS(CE*UE),ABS(CW*UW),
     &                 ABS(CT*WT),ABS(CB*WB),abs(spm(i,j,k)))
               END IF
 100        CONTINUE
 110     CONTINUE
 120  CONTINUE
      CONVS = CONVS/((KEND-KST+1)*(JEND-JST+1)*(IEND-IST+1))

C-----MODIFICATIONS FOR INTERIOR INLETS---------------------------------

      DO 170 K = KST,KEND
         DO 160 J = JST,JEND
            DO 150 I = IST,IEND
               IF (INCELL(1,J,K).GT.1 .AND. 
     $              I.LE.INCELL(1,J,K)) THEN
                  CALL FIXVAL(indexLow, indexHigh,
     &                 PP(1,J,K),AE,AW,AN,AS,AT,AB,SU,SP,
     &                 CONST,I,J,K)
                  AW(I+1,J,K) = ZERO
               END IF
               IF (INCELL(2,J,K).LT.NI .AND. 
     $              I.GE.INCELL(2,J,K)) THEN
                  CALL FIXVAL(indexLow, indexHigh, 
     &                 PP(NI,J,K),AE,AW,AN,AS,AT,AB,SU,SP,
     &                 CONST,I,J,K)
                  AE(I-1,J,K) =ZERO
               END IF
               IF (INCELL(3,I,K).GT.1 .AND. 
     $              J.LE.INCELL(3,I,K)) THEN
                  CALL FIXVAL(indexLow, indexHigh, 
     &                 PP(I,1,K),AE,AW,AN,AS,AT,AB,
     $                 SU,SP,CONST,I,J,K)
                  AS(I,J+1,K) = ZERO
               END IF
               IF (INCELL(4,I,K).LT.NJ .AND. 
     $              J.GE.INCELL(4,I,K)) THEN
                  CALL FIXVAL(indexLow, indexHigh,
     &                 PP(I,NJ,K),AE,AW,AN,AS,AT,AB,
     $                 SU,SP,CONST,I,J,K)
                  AN(I,J-1,K) = ZERO
               END IF
               IF (INCELL(5,I,J).GT.1 .AND. 
     $              K.LE.INCELL(5,I,J)) THEN
                  CALL FIXVAL(indexLow, indexHigh,
     &                 PP(I,J,1),AE,AW,AN,AS,AT,AB,
     $                 SU,SP,CONST,I,J,K)
                  AB(I,J,K+1) = ZERO
               END IF
               IF (INCELL(6,I,J).LT.NK .AND. 
     $              K.GE.INCELL(6,I,J)) THEN
                  CALL FIXVAL(indexLow, indexHigh,
     &                 PP(I,J,NK),AE,AW,AN,AS,AT,AB,
     $                 SU,SP,CONST,I,J,K)
                  AT(I,J,K-1) = ZERO
               END IF
 150        CONTINUE
 160     CONTINUE
 170  CONTINUE

C-----BOUNDARY CONDITIONS----------------------------------------------

      CALL BCP(indexLow, indexHigh, NI, NJ, NK, NITER, PCELL,
     &     PP,
     &     AE,AW,AN,AS,AT,AB,
     &     RR,
     $     LWRAP,LSYMB,LPRDKK, LPRDAV, LPRDJK)

C-----FINAL COEFFICIENT ASSEMBLY----------------------------------------

      CALL APCAL(indexLow, indexHigh, 
     &     AP,AE,AW,AN,AS,AT,AB,SP)

C-----COMPUTE THE RESIDUAL OF THE PRESSURE EQUATION---------------------
      
      IF (LEXCP.OR.(IMODE.NE.2)) THEN
         CALL RESCAL(indexLow, indexHigh,
     &        PP,RESP,
     &        AE,AW,AN,AS,AT,AB,AP,SU,
     &        RESORP)
         CALL OMGCAL(indexLow, indexHigh,
     &        PP,SCF,RESP,
     &        AE,AW,AN,AS,AT,AB,AP,SU,
     $        TRNCP,RWL1,RWL2,RWINF,
     $        IINF,JINF,KINF,IEQNIN,NTOTAL,
     $        IEQN,NITER,NIJK,IK,IJ,
     $        LCSCF)
         RESRP0 = RESORP
         TRNCP0 = TRNCP
      ENDIF

C-----STOP IF IMODE=0---------------------------------------------------

      IF (.NOT.LEXCP.AND.(IMODE.EQ.0)) GOTO 999

      IF (LPCON) GOTO 999

C-----SOLVE EQUATION SYSTEM---------------------------------------------

      RESORP = RESRP0
      TRNCP  = TRNCP0

      IF (ISOPT.EQ.1) R_NORM = TRNCP
      IF (ISOPT.EQ.3) THEN
         IF (LLEFT) R_NORM = ONE
         IF (.NOT.LLEFT) R_NORM = TRNCP
      ENDIF

      CALL SOLVE(indexLow, indexHigh,
     &     PP,PRCOR,
     &     AE,AW,AN,AS,AT,AB,AP,SU,
     $     RESP,
     &     GAMP,URFP,TSOLV,THETA,CONVL,R_NORM,
     $     NI,NJ,NK,
     &     ILCNT,IEQN,ISPP,NSWPP,ISOPT,
     $     .TRUE.,.TRUE.,.TRUE.,LPAR,LTEST,LLEFT,
     &     tmpXYZArr1, tmpXYZArr2,
     &     tmpXArr1, tmpXArr2,
     $     NEQ, CLABEL)

      IF (CONVL*TRNCP.NE.ZERO .AND. CONVS*TRNCP.NE.ZERO) THEN
         RESORP = MIN(CONVL*TRNCP,CONVS*TRNCP,RESORP)
         RESRP0 = MIN(CONVL*TRNCP,CONVS*TRNCP,RESRP0)
      ENDIF

*-----Reference pressure and periodic boundary conditions for
*-----immediate update case

      IF (.NOT.LPAR) THEN
         PPREF = PP(IPREF,JPREF,KPREF)
         DO 520 K = KST,KEND
            DO 510 J = JST,JEND
               DO 500 I = IST,IEND 
                  IF (PCELL(I,J,K).EQ.FFIELD) THEN
                     PP(I,J,K) = PP(I,J,K)-PPREF
                     PP(I,J,K) = PP(I,J,K) + PFIX
                  ENDIF
 500           CONTINUE
 510        CONTINUE
 520     CONTINUE
         IF (LWRAP)
     $        CALL PRDBC1(indexLow, indexHigh, NITER, PCELL,
     $        PP,RR,
     &        LPRDAV, LPRDJK, LPRDKK)
      ENDIF

*----------------------------------------------------------------------- 

 999  RETURN
      END
      

c*********************************************************************
c
c Revision 1.1  2000/04/12 20:31:48  rawat
c modified PressureSolver and added fortran subroutines
c
c
c*********************************************************************
