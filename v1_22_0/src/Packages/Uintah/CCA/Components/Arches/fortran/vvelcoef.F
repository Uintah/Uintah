c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/vvelcoef_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for vvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTV, JSTV, KSTV, IENDV, JENDV, KENDV
      integer i, j, k

      logical*1 L2UP, LSMPT, LCENTRAL, filtcend

      double precision AREAN, AREAS, AREAEW, AREATB, VOL
      double precision GE, GP, GW, GN, GSE, GS, GSW, GT, GST, GSB, GB
      double precision GSP, GNP
      double precision CEO, CWO, CNO, CSO, CTO, CBO
      double precision CE, CW, CN, CS, CT, CB
      double precision GUP, CPE, CPW, CPN, CPS, GWP, CPT, CPB
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DEO, DWO, DNO, DSO, DTO, DBO
      double precision DE, DW, DN, DS, DT, DB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision TEW, TNS, TTB
      double precision AEC, AWC, ANC, ASC, ATC, ABC

c*********************************************************************
c     Start :
c*********************************************************************
      ISTV = idxLoV(1)
      JSTV = idxLoV(2)
      KSTV = idxLoV(3)
      IENDV = idxHiV(1)
      JENDV = idxHiV(2)
      KENDV = idxHiV(3)

c*********************************************************************
c     Set all flags except one to false (** WARNING ** need to change)
c*********************************************************************
      LSMPT = .FALSE.
      LCENTRAL = .FALSE.
      filtcend = .false.
      L2UP = .FALSE.
      

c*********************************************************************
c     Go thru all the cells
c*********************************************************************
      DO 220 K = KSTV,KENDV
         DO 210 J = JSTV,JENDV
            DO 200 I = ISTV,IENDV
C-----------------------------------------------------------------------
C     COMPUTE AREAS AND VOLUME
C-----------------------------------------------------------------------
c              RCVA = 0.5*(RCV(J)+RCV(J+1))
c              AREAN = RCV(J+1)*STB(K)*SEW(I)
c              AREAS = RCV(J)*STB(K)*SEW(I)
c              AREAEW = RCVA*STB(K)*SNSV(J)
c              AREATB = SNSV(J)*SEW(I)
c              VOL = RCVA*STB(K)*SEW(I)*SNSV(J)
               AREAN = STB(K)*SEW(I)
               AREAS = STB(K)*SEW(I)
               AREAEW = STB(K)*SNSV(J)
               AREATB = SNSV(J)*SEW(I)
               VOL = STB(K)*SEW(I)*SNSV(J)
C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = 0.5D0*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GSE = 0.5D0*(DEN(I,J-1,K)+DEN(I+1,J-1,K))*UU(I+1,J-1,K)
               GW = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))*UU(I,J,K)
               GSW = 0.5D0*(DEN(I,J-1,K)+DEN(I-1,J-1,K))*UU(I,J-1,K)
               GN = 0.5D0*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GP = 0.5D0*(DEN(I,J,K)+DEN(I,J-1,K))*VV(I,J,K)
               GS = 0.5D0*(DEN(I,J-1,K)+DEN(I,J-2,K))*VV(I,J-1,K)
c               GN = DEN(I,J,K)*(VV(I,J+1,K)+VV(I,J,K))
c               GP = 0.0D0
c               GS = DEN(I,J-1,K)*(VV(I,J,K)+VV(I,J-1,K))
c******************
c ** WARNING ** RAWAT : What does J==2 mean in the patch context ?
c******************
c done for periodic...comment out for the time being
c               IF (J.EQ.2) GS = DEN(I,J-1,K)*VV(I,J-1,K)
               GT = 0.5D0*(DEN(I,J,K+1)+DEN(I,J,K))*WW(I,J,K+1)
               GST = 0.5D0*(DEN(I,J-1,K+1)+DEN(I,J-1,K))*WW(I,J-1,K+1)
               GB = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
               GSB = 0.5D0*(DEN(I,J-1,K)+DEN(I,J-1,K-1))*WW(I,J-1,K)
               CEO = (0.5D0*GE+0.5D0*GSE)*AREAEW
               CWO = (0.5D0*GW+0.5D0*GSW)*AREAEW
               CTO = (0.5D0*GT+0.5D0*GST)*AREATB
               CBO = (0.5D0*GB+0.5D0*GSB)*AREATB 
               CNO = 0.5D0*(GN+GP)*AREAN
               CSO = 0.5D0*(GP+GS)*AREAS
C-----------------------------------------------------------------------
C      NEW DIFFERENCING STUFF
C-----------------------------------------------------------------------
               CE = CEE(I)*CEO + CWE(I)*CWO
               CW = CWW(I)*CWO - CWE(I)*CEO   
               CN = CNNV(J)*CNO + CSNV(J)*CSO
               CS = CSSV(J)*CSO - CSNV(J)*CNO
               CT = CTT(K)*CTO + CBT(K)*CBO
               CB = CBB(K)*CBO - CBT(K)*CTO
               CESAV(I,J,K) = CE
               CWSAV(I,J,K) = CW
               CNSAV(I,J,K) = CN
               CSSAV(I,J,K) = CS
               CTSAV(I,J,K) = CT
               CBSAV(I,J,K) = CB


c modified central differencing
               if (LCENTRAL) then
                  GE = 0.5D0*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
                 GSE = 0.5D0*(DEN(I,J-1,K)+DEN(I+1,J-1,K))*UU(I+1,J-1,K)
                  GW = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))*UU(I,J,K)
                  GSW = 0.5D0*(DEN(I,J-1,K)+DEN(I-1,J-1,K))*UU(I,J-1,K)
                  GN = DEN(I,J,K)*VV(I,J+1,K)
                  GNP = DEN(I,J,K)*VV(I,J,K)
                  GS = DEN(I,J-1,K)*VV(I,J-1,K)
                  GSP = DEN(I,J-1,K)*VV(I,J,K)
                  GT = 0.5D0*(DEN(I,J,K+1)+DEN(I,J,K))*WW(I,J,K+1)
                 GST = 0.5D0*(DEN(I,J-1,K+1)+DEN(I,J-1,K))*WW(I,J-1,K+1)
                  GB = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
                  GSB = 0.5D0*(DEN(I,J-1,K)+DEN(I,J-1,K-1))*WW(I,J-1,K)
                  CEO = (0.5D0*GE+0.5D0*GSE)*AREAEW
                  CWO = (0.5D0*GW+0.5D0*GSW)*AREAEW
                  CTO = (0.5D0*GT+0.5D0*GST)*AREATB
                  CBO = (0.5D0*GB+0.5D0*GSB)*AREATB 
                  CNO = 0.5D0*(GN+GNP)*AREAN
                  CSO = 0.5D0*(GSP+GS)*AREAS
C-----------------------------------------------------------------------
C      NEW DIFFERENCING STUFF
C-----------------------------------------------------------------------
                  CE = CEO 
                  CW = CWO
                  CN = CNO
                  CS = CSO
                  CT = CTO 
                  CB = CBO
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
         GUP = EFAC(I)*(0.5D0*GE+0.5D0*GSE)+WFAC(I)*(0.5D0*GW+0.5D0*GSW)
               CPE = GUP*VOL/DXEP(I)
               CPW = GUP*VOL/DXPW(I)
               CPN = GP*VOL/SNS(J)
               CPS = GP*VOL/SNS(J-1)
         GWP = TFAC(K)*(0.5D0*GT+0.5D0*GST)+BFAC(K)*(0.5D0*GB+0.5D0*GSB)
c              CPT = GWP*VOL/(RCVA*DZTP(K))
c              CPB = GWP*VOL/(RCVA*DZPB(K))
               CPT = GWP*VOL/DZTP(K)
               CPB = GWP*VOL/DZPB(K)
C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I+1,J,K))+
     &              0.5D0*(VIS(I,J-1,K)+VIS(I+1,J-1,K)))
               VISW = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I-1,J,K))+
     &              0.5D0*(VIS(I,J-1,K)+VIS(I-1,J-1,K)))
c******************
c ** WARNING ** RAWAT : Why are VISN, VISS, VIST, VISB arrays in vcoef.f
c                       but VISE, VISW are not arrays in ucoef.f ?
c******************
c              VISN(I) = FAC1V(J)*VIS(I,J,K)+FAC2V(J)*VIS(I,J+JNSDV(J),K)
c              VISS(I) = FAC3V(J)*VIS(I,J-1,K)+FAC4V(J)*VIS(I,J+JSSDV(J),K)  
c              VIST(I) = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K+1))+
c    &              0.5D0*(VIS(I,J-1,K)+VIS(I,J-1,K+1)))
c              VISB(I) = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K-1))+
c    &              0.5D0*(VIS(I,J-1,K)+VIS(I,J-1,K-1)))
               VISN = FAC1V(J)*VIS(I,J,K)+FAC2V(J)*VIS(I,J+JNSDV(J),K)
               VISS = FAC3V(J)*VIS(I,J-1,K)+FAC4V(J)*VIS(I,J+JSSDV(J),K)
               VIST = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K+1))+
     &              0.5D0*(VIS(I,J-1,K)+VIS(I,J-1,K+1)))
               VISB = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J,K-1))+
     &              0.5D0*(VIS(I,J-1,K)+VIS(I,J-1,K-1)))
               DEO = VISE*AREAEW/DXEP(I)
               DWO = VISW*AREAEW/DXPW(I)
c              DNO = VISN(I)*AREAN/DYNPV(J)
c              DSO = VISS(I)*AREAS/DYPSV(J)
c              DTO = VIST(I)*AREATB/(RCVA*DZTP(K))
c              DBO = VISB(I)*AREATB/(RCVA*DZPB(K))
               DNO = VISN*AREAN/DYNPV(J)
               DSO = VISS*AREAS/DYPSV(J)
               DTO = VIST*AREATB/DZTP(K)
               DBO = VISB*AREATB/DZPB(K)
C-----------------------------------------------------------------------
C     NEW DIFFUSION TERMS
C-----------------------------------------------------------------------
               DE = CEE(I)*DEO + CWE(I)*DWO*DXPW(I)/DXEP(I)
               DW = CWW(I)*DWO - CWE(I)*DEO*DXEP(I)/DXPW(I)   
               DN = CNNV(J)*DNO + CSNV(J)*DSO*DYPSV(J)/DYNPV(J)
               DS = CSSV(J)*DSO - CSNV(J)*DNO*DYNPV(J)/DYPSV(J)
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
c ** This explicit double precision stuff is for the aix
               AEC = -0.5D0*CE*TEW + dMAX1(0.0D0,-CPE)*(1.0D0-TEW)
               AWC = 0.5D0*CW*TEW + dMAX1(0.0D0,CPW)*(1.0D0-TEW)
               ANC = -0.5D0*CN*TNS + dMAX1(0.0D0,-CPN)*(1.0D0-TNS)
               ASC = 0.5D0*CS*TNS + dMAX1(0.0D0,CPS)*(1.0D0-TNS)
               ATC = -0.5D0*CT*TTB + dMAX1(0.0D0,-CPT)*(1.0D0-TTB)
               ABC = 0.5D0*CB*TTB + dMAX1(0.0D0,CPB)*(1.0D0-TTB)
c               DV(I,J,K) = VOL/DYPS(J)

C----------------------------------------------------------------------
C      PROVISIONS FOR SIMPLEST
C----------------------------------------------------------------------
               IF (LSMPT) THEN
                  AE(I,J,K) = DE
                  AW(I,J,K) = DW
                  AN(I,J,K) = DN
                  AS(I,J,K) = DS
                  AT(I,J,K) = DT
                  AB(I,J,K) = DB
c                 AC(I,J,K) = AEC+AWC+ANC+ASC+ATC+ABC
c                 SU(I,J,K) = AEC*V(I+1,J,K)+AWC*V(I-1,J,K) +
c    &                 ANC*V(I,J+1,K)+ASC*V(I,J-1,K) +
c    &                 ATC*V(I,J,K+1)+ABC*V(I,J,K-1)
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
c Revision 1.7  2000/10/08 18:56:41  rawat
c fixed the solver for multi
c
c Revision 1.6  2000/09/28 22:16:47  bigler
c Changed certain functions to explicity be double precision functions.
c
c Revision 1.5  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.4  2000/08/10 21:29:14  rawat
c fixed a bug in cellinformation
c
c Revision 1.3  2000/08/10 00:56:38  rawat
c added pressure bc for scalar and changed discretization option for velocity
c
c Revision 1.2  2000/07/14 03:45:48  rawat
c completed velocity bc and fixed some bugs
c
c Revision 1.1  2000/07/08 23:08:56  bbanerje
c Added vvelcoef and wvelcoef ..
c Rawat check the ** WARNING ** tags in these files for possible problems.
c
c
c*********************************************************************
