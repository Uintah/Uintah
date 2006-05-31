c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/uvelcoef_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTU, JSTU, KSTU, IENDU, JENDU, KENDU
      integer i, j, k

      logical*1 L2UP, LSMPT, LCENTRAL, filtcend

      double precision AREAN, AREAS, AREAEW, AREATB, VOL
      double precision GE, GP, GW, GN, GNW, GS, GSW, GT, GTW, GTP, GBW
      double precision CEO, CWO, CNO, CSO, CTO, CBO
      double precision CE, CW, CN, CS, CT, CB
      double precision CPE, CPW, GVP, CPN, CPS, CPT, CPB
      double precision CP, GEP, GWP
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

      LSMPT = .FALSE.
      LCENTRAL = .FALSE.
      filtcend = .false.
      L2UP = .false.
c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      DO 220 K = KSTU,KENDU
         DO 210 J = JSTU,JENDU
            DO 200 I = ISTU,IENDU   
C-----------------------------------------------------------------------
C     COMPUTE AREAS AND VOLUME (** WARNING ** No polar co-ordinate systems)
C-----------------------------------------------------------------------
C              AREAN = RV(J+1)*SEWU(I)*STB(K)
C              AREAS = RV(J)*SEWU(I)*STB(K)
C              AREAEW = RA(J)*SNS(J)*STB(K)
C              AREATB = SNS(J)*SEWU(I)
C              VOL = RA(J)*SEWU(I)*SNS(J)*STB(K)
               AREAN = SEWU(I)*STB(K)
               AREAS = SEWU(I)*STB(K)
               AREAEW = SNS(J)*STB(K)
               AREATB = SNS(J)*SEWU(I)
               VOL = SEWU(I)*SNS(J)*STB(K)
C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = 0.5D0*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GP = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))*UU(I,J,K)
               GW = 0.5D0*(DEN(I-1,J,K)+DEN(I-2,J,K))*UU(I-1,J,K)
c               GE = DEN(I,J,K)*(UU(I+1,J,K)+UU(I,J,K))
c               GP = 0.0D0
c               GW = DEN(I-1,J,K)*(UU(I,J,K)+UU(I-1,J,K))
               GN = 0.5D0*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GNW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J+1,K))*VV(I-1,J+1,K)
               GS = 0.5D0*(DEN(I,J-1,K)+DEN(I,J,K))*VV(I,J,K)
               GSW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J-1,K))*VV(I-1,J,K)
               GT = 0.5D0*(DEN(I,J,K)+DEN(I,J,K+1))*WW(I,J,K+1)
               GTW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J,K+1))*WW(I-1,J,K+1)
               GTP = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
               GBW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J,K-1))*WW(I-1,J,K)
               CEO = 0.5D0*(GE+GP)*AREAEW
               CWO = 0.5D0*(GP+GW)*AREAEW
               CNO = (0.5D0*GN+0.5D0*GNW)*AREAN
               CSO = (0.5D0*GS+0.5D0*GSW)*AREAS
               CTO = (0.5D0*GT+0.5D0*GTW)*AREATB
               CBO = (0.5D0*GTP+0.5D0*GBW)*AREATB
C-----------------------------------------------------------------------
C      NEW DIFFERENCING STUFF
C-----------------------------------------------------------------------
               CE = CEEU(I)*CEO + CWEU(I)*CWO
               CW = CWWU(I)*CWO - CWEU(I)*CEO   
               CN = CNN(J)*CNO + CSN(J)*CSO
               CS = CSS(J)*CSO - CSN(J)*CNO
               CT = CTT(K)*CTO + CBT(K)*CBO
               CB = CBB(K)*CBO - CBT(K)*CTO
               CESAV(I,J,K) = CE
               CWSAV(I,J,K) = CW
               CNSAV(I,J,K) = CN
               CSSAV(I,J,K) = CS
               CTSAV(I,J,K) = CT
               CBSAV(I,J,K) = CB
c recompute central differencing for explicit solvers
               if (LCENTRAL) then
                  GE = DEN(I,J,K)*UU(I+1,J,K)
                  GEP = DEN(I,J,K)*UU(I,J,K)
c     GP = 0.5*(DEN(I,J,K)+DEN(I-1,J,K))*UU(I,J,K)
                  GW = DEN(I-1,J,K)*UU(I-1,J,K)
                  GWP = DEN(I-1,J,K)*UU(I,J,K)
                  GN = 0.5D0*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
                 GNW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J+1,K))*VV(I-1,J+1,K)
                  GS = 0.5D0*(DEN(I,J-1,K)+DEN(I,J,K))*VV(I,J,K)
                  GSW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J-1,K))*VV(I-1,J,K)
                  GT = 0.5D0*(DEN(I,J,K)+DEN(I,J,K+1))*WW(I,J,K+1)
                 GTW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J,K+1))*WW(I-1,J,K+1)
                  GTP = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
                  GBW = 0.5D0*(DEN(I-1,J,K)+DEN(I-1,J,K-1))*WW(I-1,J,K)
                  CEO = 0.5D0*(GE+GEP)*AREAEW
                  CWO = 0.5D0*(GWP+GW)*AREAEW
                  CNO = (0.5D0*GN+0.5D0*GNW)*AREAN
                  CSO = (0.5D0*GS+0.5D0*GSW)*AREAS
                  CTO = (0.5D0*GT+0.5D0*GTW)*AREATB
                  CBO = (0.5D0*GTP+0.5D0*GBW)*AREATB
C-----------------------------------------------------------------------
C     NEW DIFFERENCING STUFF
C-----------------------------------------------------------------------
                  CE = CEO 
                  CW = CWO 
                  CN = CNO
                  CS = CSO
                  CT = CTO
                  CB = CBO
                  CP = CEO-CWO+CNO-CSO+CTO-CBO
                  CESAV(I,J,K) = CE
                  CWSAV(I,J,K) = CW
                  CNSAV(I,J,K) = CN
                  CSSAV(I,J,K) = CS
                  CTSAV(I,J,K) = CT
                  CBSAV(I,J,K) = CB
               endif

C-----------------------------------------------------------------------
C      UPWIND DIFFERENCING STUFF
C-----------------------------------------------------------------------
               CPE = GP*VOL/SEW(I)
               CPW = GP*VOL/SEW(I-1)
        GVP = ENFAC(J)*(0.5D0*GN+0.5D0*GNW)+SFAC(J)*(0.5D0*GS+0.5D0*GSW)
               CPN = GVP*VOL/DYNP(J)
               CPS = GVP*VOL/DYPS(J)
        GWP = TFAC(K)*(0.5D0*GT+0.5D0*GTW)+BFAC(K)*(0.5D0*GTP+0.5D0*GBW)
c              CPT = GWP*VOL/(RA(J)*DZTP(K))
c              CPB = GWP*VOL/(RA(J)*DZPB(K))
               CPT = GWP*VOL/DZTP(K)
               CPB = GWP*VOL/DZPB(K)
C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = FAC1U(I)*VIS(I,J,K)+FAC2U(I)*VIS(I+IESDU(I),J,K)
               VISW = FAC3U(I)*VIS(I-1,J,K)+FAC4U(I)*VIS(I+IWSDU(I),J,K)
               VISN = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J+1,K))+
     &              0.5D0*(VIS(I-1,J,K)+VIS(I-1,J+1,K)))
               VISS = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J-1,K))+
     &              0.5D0*(VIS(I-1,J,K)+VIS(I-1,J-1,K)))
               VIST = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K+1))+
     &              0.5D0*(VIS(I-1,J,K)+VIS(I-1,J,K+1)))
               VISB = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K-1))+
     &              0.5D0*(VIS(I-1,J,K)+VIS(I-1,J,K-1)))  
               DEO = VISE*AREAEW/DXEPU(I)
               DWO = VISW*AREAEW/DXPWU(I)
               DNO = VISN*AREAN/DYNP(J)
               DSO = VISS*AREAS/DYPS(J)
c              DTO = VIST*AREATB/(RA(J)*DZTP(K))
c              DBO = VISB*AREATB/(RA(J)*DZPB(K))
               DTO = VIST*AREATB/DZTP(K)
               DBO = VISB*AREATB/DZPB(K)
C-----------------------------------------------------------------------
C     NEW DIFFUSION TERMS
C-----------------------------------------------------------------------
               DE = CEEU(I)*DEO + CWEU(I)*DWO*DXPWU(I)/DXEPU(I)
               DW = CWWU(I)*DWO - CWEU(I)*DEO*DXEPU(I)/DXPWU(I)   
               DN = CNN(J)*DNO + CSN(J)*DSO*DYPS(J)/DYNP(J)
               DS = CSS(J)*DSO - CSN(J)*DNO*DYNP(J)/DYPS(J)
               DT = CTT(K)*DTO + CBT(K)*DBO*DZPB(K)/DZTP(K)
               DB = CBB(K)*DBO - CBT(K)*DTO*DZTP(K)/DZPB(K)
C-----------------------------------------------------------------------
C     CONVECTION DIFFERENCING SCHEME SELECTION
C-----------------------------------------------------------------------
               COEFE = DE - 0.5D0*ABS(CE)
               COEFW = DW - 0.5D0*ABS(CW)
               COEFN = DN - 0.5D0*ABS(CN)
               COEFS = DS - 0.5D0*ABS(CS)
               COEFT = DT - 0.5D0*ABS(CT)
               COEFB = DB - 0.5D0*ABS(CB)
               TEW = 1.0D0
               TNS = 1.0D0
               TTB = 1.0D0
               IF (COEFE.LT.0.0D0 .OR. COEFW.LT.0.0D0) TEW = 0.0D0
               IF (COEFN.LT.0.0D0 .OR. COEFS.LT.0.0D0) TNS = 0.0D0
               IF (COEFT.LT.0.0D0 .OR. COEFB.LT.0.0D0) TTB = 0.0D0
c               if (tew*tns*ttb.eq.0.0) write(6,*) i,' ',j,' ',k
c ** this is a version that uses double precision
               AEC = -0.5D0*CE*TEW + dMAX1(0.0D0,-CPE)*(1.0D0-TEW)
               AWC = 0.5D0*CW*TEW + dMAX1(0.0D0,CPW)*(1.0D0-TEW)
               ANC = -0.5D0*CN*TNS + dMAX1(0.0D0,-CPN)*(1.0D0-TNS)
               ASC = 0.5D0*CS*TNS + dMAX1(0.0D0,CPS)*(1.0D0-TNS)
               ATC = -0.5D0*CT*TTB + dMAX1(0.0D0,-CPT)*(1.0D0-TTB)
               ABC = 0.5D0*CB*TTB + dMAX1(0.0D0,CPB)*(1.0D0-TTB)
c              TSUM = TSUM + TEW+TNS+TTB
C----------------------------------------------------------------------
C      PROVISIONS FOR VARIOUS UPWIND DIFFERENCING SCHEMES
C----------------------------------------------------------------------
               IF (LSMPT) THEN
                  AE(I,J,K) = DE
                  AW(I,J,K) = DW
                  AN(I,J,K) = DN
                  AS(I,J,K) = DS
                  AT(I,J,K) = DT
                  AB(I,J,K) = DB
c                 AC(I,J,K) = AEC+AWC+ANC+ASC+ATC+ABC
c                 SU(I,J,K) = AEC*U(I+1,J,K)+AWC*U(I-1,J,K) +
c    &                 ANC*U(I,J+1,K)+ASC*U(I,J-1,K) +
c    &                 ATC*U(I,J,K+1)+ABC*U(I,J,K-1)
               ELSE IF (LCEND) THEN
                  AE(I,J,K) = DE-CE*0.5D0
                  AW(I,J,K) = DW+CW*0.5D0
                  AN(I,J,K) = DN-CN*0.5D0
                  AS(I,J,K) = DS+CS*0.5D0
                  AT(I,J,K) = DT-CT*0.5D0
                  AB(I,J,K) = DB+CB*0.5D0
c                 AC(I,J,K) = 0.0
c                 SU(I,J,K) = 0.0
               ELSE IF (LCENTRAL) THEN
                  AE(I,J,K) = DE
                  AW(I,J,K) = DW
                  AN(I,J,K) = DN
                  AS(I,J,K) = DS
                  AT(I,J,K) = DT
                  AB(I,J,K) = DB
c                 AC(I,J,K) = 0.0
c                 SU(I,J,K) = 0.0
               ELSE IF (FILTCEND) THEN
                  AE(I,J,K) = DE
                  AW(I,J,K) = DW
                  AN(I,J,K) = DN
                  AS(I,J,K) = DS
                  AT(I,J,K) = DT
                  AB(I,J,K) = DB
                  CESAV(I,J,K) = 0.0D0
                  CWSAV(I,J,K) = 0.0D0
                  CNSAV(I,J,K) = 0.0D0
                  CSSAV(I,J,K) = 0.0D0
                  CTSAV(I,J,K) = 0.0D0
                  CBSAV(I,J,K) = 0.0D0
               ELSE IF (L2UP) THEN
                  AE(I,J,K) = DE*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFE)))
     &                 + dMAX1(0.0D0,COEFE) + dMAX1(0.0D0,-CE)
                  AW(I,J,K) = DW*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFW)))
     &                 + dMAX1(0.0D0,COEFW) + dMAX1(0.0D0,CW)
                  AN(I,J,K) = DN*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFN)))
     &                 + dMAX1(0.0D0,COEFN) + dMAX1(0.0D0,-CN)
                  AS(I,J,K) = DS*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFS)))
     &                 + dMAX1(0.0D0,COEFS) + dMAX1(0.0D0,CS)
                  AT(I,J,K) = DT*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFT)))
     &                 + dMAX1(0.0D0,COEFT) + dMAX1(0.0D0,-CT)
                  AB(I,J,K) = DB*(1.0D0-dMAX1(0.0D0,dSIGN(1.0D0,COEFB)))
     &                 + dMAX1(0.0D0,COEFB) + dMAX1(0.0D0,CB)
c                 AC(I,J,K) = 0.0
c                 SU(I,J,K) = 0.0
               ELSE
                  AE(I,J,K) = AEC + DE
                  AW(I,J,K) = AWC + DW
                  AN(I,J,K) = ANC + DN
                  AS(I,J,K) = ASC + DS
                  AT(I,J,K) = ATC + DT
                  AB(I,J,K) = ABC + DB
c                 AC(I,J,K) = 0.0
c                 SU(I,J,K) = 0.0
               END IF
 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

c*********************************************************************
c
c Revision 1.9  2000/10/08 18:56:41  rawat
c fixed the solver for multi
c
c Revision 1.8  2000/09/28 22:16:47  bigler
c Changed certain functions to explicity be double precision functions.
c
c Revision 1.7  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.6  2000/08/10 21:29:14  rawat
c fixed a bug in cellinformation
c
c Revision 1.5  2000/08/10 00:56:38  rawat
c added pressure bc for scalar and changed discretization option for velocity
c
c Revision 1.4  2000/07/14 03:45:48  rawat
c completed velocity bc and fixed some bugs
c
c Revision 1.3  2000/07/08 23:08:56  bbanerje
c Added vvelcoef and wvelcoef ..
c Rawat check the ** WARNING ** tags in these files for possible problems.
c
c Revision 1.2  2000/07/08 08:03:37  bbanerje
c Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
c made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
c to Arches::AE .. doesn't like enums in templates apparently.
c
c Revision 1.1  2000/07/07 23:07:48  rawat
c added inlet bc's
c
c*********************************************************************
