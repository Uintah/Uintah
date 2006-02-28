c*********************************************************************
c
c
c*********************************************************************

#include <Packages/Uintah/CCA/Components/Arches/fortran/wvelcoef_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTW, JSTW, KSTW, IENDW, JENDW, KENDW
      integer i, j, k

      logical*1 L2UP, LSMPT, LCENTRAL, filtcend

      double precision AREAN, AREAS, AREAEW, AREATB, VOL
      double precision GE, GEB, GW, GWB, GN, GNB, GS, GSB, GT, GPB, GB
      double precision GTP, GBP
      double precision CEO, CWO, CNO, CSO, CTO, CBO
      double precision CE, CW, CN, CS, CT, CB
      double precision GUP, CPE, CPW, GVP, CPN, CPS, CPT, CPB
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DEO, DWO, DNO, DSO, DTO, DBO
      double precision DE, DW, DN, DS, DT, DB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision TEW, TNS, TTB
      double precision AEC, AWC, ANC, ASC, ATC, ABC

c*********************************************************************
c     Start :
c*********************************************************************
      ISTW = idxLoW(1)
      JSTW = idxLoW(2)
      KSTW = idxLoW(3)
      IENDW = idxHiW(1)
      JENDW = idxHiW(2)
      KENDW = idxHiW(3)

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
      DO 220 K = KSTW,KENDW
         DO 210 J = JSTW,JENDW
            DO 200 I = ISTW,IENDW   
C-----------------------------------------------------------------------
C     COMPUTE AREAS AND VOLUME
C-----------------------------------------------------------------------
c              AREAN = RV(J+1)*SEW(I)*STBW(K)
c              AREAS = RV(J)*SEW(I)*STBW(K)
c              AREAEW = RA(J)*SNS(J)*STBW(K)
c              AREATB = SNS(J)*SEW(I)
c              VOL = RA(J)*SEW(I)*SNS(J)*STBW(K)
               AREAN = SEW(I)*STBW(K)
               AREAS = SEW(I)*STBW(K)
               AREAEW = SNS(J)*STBW(K)
               AREATB = SNS(J)*SEW(I)
               VOL = SEW(I)*SNS(J)*STBW(K)
C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               GE = 0.5D0*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
               GEB = 0.5D0*(DEN(I+1,J,K-1)+DEN(I,J,K-1))*UU(I+1,J,K-1)
               GW = 0.5D0*(DEN(I-1,J,K)+DEN(I,J,K))*UU(I,J,K)
               GWB = 0.5D0*(DEN(I,J,K-1)+DEN(I-1,J,K-1))*UU(I,J,K-1)
               GN = 0.5D0*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
               GNB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J+1,K-1))*VV(I,J+1,K-1)
               GS = 0.5D0*(DEN(I,J-1,K)+DEN(I,J,K))*VV(I,J,K)
               GSB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J-1,K-1))*VV(I,J,K-1)
               GT = 0.5D0*(DEN(I,J,K)+DEN(I,J,K+1))*WW(I,J,K+1)
               GPB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J,K))*WW(I,J,K)
               GB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J,K-2))*WW(I,J,K-1)
c ****
c ** WARNING ** Rawat, what does k == 2 mean in this context ?
c ****
c used for perodic bc               
c               IF (K.EQ.2) GB = DEN(I,J,K-1)*WW(I,J,K-1)
               CEO = (0.5D0*GE+0.5D0*GEB)*AREAEW
               CWO = (0.5D0*GW+0.5D0*GWB)*AREAEW
               CNO = (0.5D0*GN+0.5D0*GNB)*AREAN
               CSO = (0.5D0*GS+0.5D0*GSB)*AREAS
               CTO = 0.5D0*(GT+GPB)*AREATB
               CBO = 0.5D0*(GPB+GB)*AREATB
C-----------------------------------------------------------------------
C      NEW DIFFERENCING STUFF
C-----------------------------------------------------------------------
               CE = CEE(I)*CEO + CWE(I)*CWO
               CW = CWW(I)*CWO - CWE(I)*CEO   
               CN = CNN(J)*CNO + CSN(J)*CSO
               CS = CSS(J)*CSO - CSN(J)*CNO
               CT = CTTW(K)*CTO + CBTW(K)*CBO
               CB = CBBW(K)*CBO - CBTW(K)*CTO
               CESAV(I,J,K) = CE
               CWSAV(I,J,K) = CW
               CNSAV(I,J,K) = CN
               CSSAV(I,J,K) = CS
               CTSAV(I,J,K) = CT
               CBSAV(I,J,K) = CB

               if (LCENTRAL) then
                  GE = 0.5D0*(DEN(I+1,J,K)+DEN(I,J,K))*UU(I+1,J,K)
                 GEB = 0.5D0*(DEN(I+1,J,K-1)+DEN(I,J,K-1))*UU(I+1,J,K-1)
                  GW = 0.5D0*(DEN(I-1,J,K)+DEN(I,J,K))*UU(I,J,K)
                  GWB = 0.5D0*(DEN(I,J,K-1)+DEN(I-1,J,K-1))*UU(I,J,K-1)
                  GN = 0.5D0*(DEN(I,J+1,K)+DEN(I,J,K))*VV(I,J+1,K)
                 GNB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J+1,K-1))*VV(I,J+1,K-1)
                  GS = 0.5D0*(DEN(I,J-1,K)+DEN(I,J,K))*VV(I,J,K)
                  GSB = 0.5D0*(DEN(I,J,K-1)+DEN(I,J-1,K-1))*VV(I,J,K-1)
                  GT = DEN(I,J,K)*WW(I,J,K+1)
                  GTP =  DEN(I,J,K)*WW(I,J,K)
                  GB = DEN(I,J,K-1)*WW(I,J,K-1)
                  GBP = DEN(I,J,K-1)*WW(I,J,K)
                  CEO = (0.5D0*GE+0.5D0*GEB)*AREAEW
                  CWO = (0.5D0*GW+0.5D0*GWB)*AREAEW
                  CNO = (0.5D0*GN+0.5D0*GNB)*AREAN
                  CSO = (0.5D0*GS+0.5D0*GSB)*AREAS
                  CTO = 0.5D0*(GT+GTP)*AREATB
                  CBO = 0.5D0*(GBP+GB)*AREATB
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
         GUP = EFAC(I)*(0.5D0*GE+0.5D0*GEB)+WFAC(I)*(0.5D0*GW+0.5D0*GWB)
               CPE = GUP*VOL/DXEP(I)
               CPW = GUP*VOL/DXPW(I)
        GVP = ENFAC(J)*(0.5D0*GN+0.5D0*GNB)+SFAC(J)*(0.5D0*GS+0.5D0*GSB)
               CPN = GVP*VOL/DYNP(J)
               CPS = GVP*VOL/DYPS(J)
c              CPT = GPB*VOL/(RA(J)*STB(K))
c              CPB = GPB*VOL/(RA(J)*STB(K-1))
               CPT = GPB*VOL/STB(K)
               CPB = GPB*VOL/STB(K-1)
C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I+1,J,K))+
     &              0.5D0*(VIS(I,J,K-1)+VIS(I+1,J,K-1)))
               VISW = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I-1,J,K))+
     &              0.5D0*(VIS(I,J,K-1)+VIS(I-1,J,K-1)))  
c ****
c ** WARNING ** Rawat, why are visn, viss, vist, visb arrays here while they
c               are not in uvelcoef ?
c ****
c              VISN(I) = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J+1,K))+
c    &              0.5D0*(VIS(I,J,K-1)+VIS(I,J+1,K-1)))
c              VISS(I) = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J-1,K))+
c    &              0.5D0*(VIS(I,J,K-1)+VIS(I,J-1,K-1)))
c              VIST(I) = FAC1W(K)*VIS(I,J,K)+FAC2W(K)*
c    &              VIS(I,J,K+KTSDW(K))
c              VISB(I) = FAC3W(K)*VIS(I,J,K-1)+FAC4W(K)*
c    &              VIS(I,J,K+KBSDW(K))
               VISN = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J+1,K))+
     &              0.5D0*(VIS(I,J,K-1)+VIS(I,J+1,K-1)))
               VISS = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J-1,K))+
     &              0.5D0*(VIS(I,J,K-1)+VIS(I,J-1,K-1)))
               VIST = FAC1W(K)*VIS(I,J,K)+FAC2W(K)*
     &              VIS(I,J,K+KTSDW(K))
               VISB = FAC3W(K)*VIS(I,J,K-1)+FAC4W(K)*
     &              VIS(I,J,K+KBSDW(K))
               DEO = VISE*AREAEW/DXEP(I)
               DWO = VISW*AREAEW/DXPW(I)
c              DNO = VISN(I)*AREAN/DYNP(J)
c              DSO = VISS(I)*AREAS/DYPS(J)
c              DTO = VIST(I)*AREATB/(RA(J)*DZTPW(K))
c              DBO = VISB(I)*AREATB/(RA(J)*DZPBW(K))
               DNO = VISN*AREAN/DYNP(J)
               DSO = VISS*AREAS/DYPS(J)
               DTO = VIST*AREATB/DZTPW(K)
               DBO = VISB*AREATB/DZPBW(K)
C-----------------------------------------------------------------------
C     NEW DIFFUSION TERMS
C-----------------------------------------------------------------------
               DE = CEE(I)*DEO + CWE(I)*DWO*DXPW(I)/DXEP(I)
               DW = CWW(I)*DWO - CWE(I)*DEO*DXEP(I)/DXPW(I)   
               DN = CNN(J)*DNO + CSN(J)*DSO*DYPS(J)/DYNP(J)
               DS = CSS(J)*DSO - CSN(J)*DNO*DYNP(J)/DYPS(J)
               DT = CTTW(K)*DTO + CBTW(K)*DBO*DZPBW(K)/DZTPW(K)
               DB = CBBW(K)*DBO - CBTW(K)*DTO*DZTPW(K)/DZPBW(K)
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
c Explicit double precision stuff for the aix
               AEC = -0.5D0*CE*TEW + dMAX1(0.0D0,-CPE)*(1.0D0-TEW)
               AWC = 0.5D0*CW*TEW + dMAX1(0.0D0,CPW)*(1.0D0-TEW)
               ANC = -0.5D0*CN*TNS + dMAX1(0.0D0,-CPN)*(1.0D0-TNS)
               ASC = 0.5D0*CS*TNS + dMAX1(0.0D0,CPS)*(1.0D0-TNS)
               ATC = -0.5D0*CT*TTB + dMAX1(0.0D0,-CPT)*(1.0D0-TTB)
               ABC = 0.5D0*CB*TTB + dMAX1(0.0D0,CPB)*(1.0D0-TTB)
c               DDW(I,J,K) = VOL/DZPB(K)

C----------------------------------------------------------------------
C      PROVISIONS FOR VARIOUS CONVECTION DIFFERENCING SCHEMES
C----------------------------------------------------------------------
               IF (LSMPT) THEN
                  AE(I,J,K) = DE
                  AW(I,J,K) = DW
                  AN(I,J,K) = DN
                  AS(I,J,K) = DS
                  AT(I,J,K) = DT
                  AB(I,J,K) = DB
c                 AC(I,J,K) = AEC+AWC+ANC+ASC+ATC+ABC
c                 SU(I,J,K) = AEC*W(I+1,J,K)+AWC*W(I-1,J,K) +
c    &                 ANC*W(I,J+1,K)+ASC*W(I,J-1,K) +
c    &                 ATC*W(I,J,K+1)+ABC*W(I,J,K-1)
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
c Revision 1.8  2000/10/08 18:56:41  rawat
c fixed the solver for multi
c
c Revision 1.7  2000/09/28 22:16:47  bigler
c Changed certain functions to explicity be double precision functions.
c
c Revision 1.6  2000/09/26 04:35:34  rawat
c added some more multi-patch support
c
c Revision 1.5  2000/08/10 21:29:15  rawat
c fixed a bug in cellinformation
c
c Revision 1.4  2000/08/10 00:56:38  rawat
c added pressure bc for scalar and changed discretization option for velocity
c
c Revision 1.3  2000/07/14 03:45:49  rawat
c completed velocity bc and fixed some bugs
c
c Revision 1.2  2000/07/08 23:42:17  bbanerje
c Changed AMAX1(A,B) to MAX(A,B)
c
c Revision 1.1  2000/07/08 23:08:56  bbanerje
c Added vvelcoef and wvelcoef ..
c Rawat check the ** WARNING ** tags in these files for possible problems.
c
c
c*********************************************************************
