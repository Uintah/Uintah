C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the \"Software\"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/wvelcoef_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for uvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTW, JSTW, KSTW, IENDW, JENDW, KENDW
      integer i, j, k

      logical*1 L2UP

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
c     Set all flags
c*********************************************************************
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

C-----------------------------------------------------------------------
C      UPWIND DIFFERENCING STUFF
C-----------------------------------------------------------------------
        GUP = EFAC(I)*(0.5D0*GE+0.5D0*GEB)+WFAC(I)*(0.5D0*GW+0.5D0*GWB)
               CPE = GUP*VOL/DXEP(I)
               CPW = GUP*VOL/DXPW(I)
        GVP = NFAC(J)*(0.5D0*GN+0.5D0*GNB)+SFAC(J)*(0.5D0*GS+0.5D0*GSB)
               CPN = GVP*VOL/DYNP(J)
               CPS = GVP*VOL/DYPS(J)
               CPT = GPB*VOL/STB(K)
               CPB = GPB*VOL/STB(K-1)
C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
c               VISE = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I+1,J,K))+
c     &              0.5D0*(VIS(I,J,K-1)+VIS(I+1,J,K-1)))
c               VISW = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I-1,J,K))+
c     &              0.5D0*(VIS(I,J,K-1)+VIS(I-1,J,K-1)))  
c               VISN = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J+1,K))+
c     &              0.5D0*(VIS(I,J,K-1)+VIS(I,J+1,K-1)))
c               VISS = 0.5D0*(0.5D0*(VIS(I,J,K)+VIS(I,J-1,K))+
c     &              0.5D0*(VIS(I,J,K-1)+VIS(I,J-1,K-1)))
               VISE = FAC1EW(I)*0.5D0*(VIS(I+1,J,K)+VIS(I+1,J,K-1))+
     +                FAC2EW(I)*0.5D0*(VIS(I+E_shift(I),J,K)+
     +                                 VIS(I+E_shift(I),J,K-1))
               VISW = FAC3EW(I)*0.5D0*(VIS(I-1,J,K)+VIS(I-1,J,K-1))+
     +                FAC4EW(I)*0.5D0*(VIS(I+W_shift(I),J,K)+
     +                                 VIS(I+W_shift(I),J,K-1))
               VISN = FAC1NS(J)*0.5D0*(VIS(I,J+1,K)+VIS(I,J+1,K-1))+
     +                FAC2NS(J)*0.5D0*(VIS(I,J+N_shift(J),K)+
     +                                 VIS(I,J+N_shift(J),K-1))
               VISS = FAC3NS(J)*0.5D0*(VIS(I,J-1,K)+VIS(I,J-1,K-1))+
     +                FAC4NS(J)*0.5D0*(VIS(I,J+S_shift(J),K)+
     +                                 VIS(I,J+S_shift(J),K-1))
               VIST = FAC1W(K)*VIS(I,J,K)+FAC2W(K)*VIS(I,J,K+KTSDW(K))
               VISB = FAC3W(K)*VIS(I,J,K-1)+FAC4W(K)*VIS(I,J,K+KBSDW(K))
               DEO = VISE*AREAEW/DXEP(I)
               DWO = VISW*AREAEW/DXPW(I)
               DNO = VISN*AREAN/DYNP(J)
               DSO = VISS*AREAS/DYPS(J)
               DTO = VIST*AREATB/DZTPW(K)
               DBO = VISB*AREATB/DZPBW(K)
C-----------------------------------------------------------------------
C     NEW DIFFUSION TERMS
C-----------------------------------------------------------------------
c viscosity interpolation for weighted central differencing is now done
c directly to faces centered about P cell center, without intermediate
c interpolation to standard P cell faces (where velocities are stored)
c               DE = CEE(I)*DEO + CWE(I)*DWO*DXPW(I)/DXEP(I)
c               DW = CWW(I)*DWO - CWE(I)*DEO*DXEP(I)/DXPW(I)   
c               DN = CNN(J)*DNO + CSN(J)*DSO*DYPS(J)/DYNP(J)
c               DS = CSS(J)*DSO - CSN(J)*DNO*DYNP(J)/DYPS(J)
c               DT = CTTW(K)*DTO + CBTW(K)*DBO*DZPBW(K)/DZTPW(K)
c               DB = CBBW(K)*DBO - CBTW(K)*DTO*DZTPW(K)/DZPBW(K)
               DE = DEO
               DW = DWO   
               DN = DNO
               DS = DSO
               DT = DTO
               DB = DBO
               if (de.lt.0.0D0) write(6,*) "DE SFCZ is negative"
               if (dw.lt.0.0D0) write(6,*) "DW SFCZ is negative"
               if (dn.lt.0.0D0) write(6,*) "DN SFCZ is negative"
               if (ds.lt.0.0D0) write(6,*) "DS SFCZ is negative"
               if (dt.lt.0.0D0) write(6,*) "DT SFCZ is negative"
               if (db.lt.0.0D0) write(6,*) "DB SFCZ is negative"
               if ((de.lt.0.0D0).or.(dw.lt.0.0D0).or.(dn.lt.0.0D0).or.
     .             (ds.lt.0.0D0).or.(dt.lt.0.0D0).or.(db.lt.0.0D0)) then
                 write(6,*) "Location indexes: ",i,j,k
                 write(6,*) "Possibly stretching is two high."
                 call exit(0)
               end if
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

C----------------------------------------------------------------------
C      PROVISIONS FOR VARIOUS CONVECTION DIFFERENCING SCHEMES
C----------------------------------------------------------------------
               IF (LCEND) THEN
                  AE(I,J,K) = DE-CE*0.5D0
                  AW(I,J,K) = DW+CW*0.5D0
                  AN(I,J,K) = DN-CN*0.5D0
                  AS(I,J,K) = DS+CS*0.5D0
                  AT(I,J,K) = DT-CT*0.5D0
                  AB(I,J,K) = DB+CB*0.5D0
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
               ELSE
                  AE(I,J,K) = AEC + DE
                  AW(I,J,K) = AWC + DW
                  AN(I,J,K) = ANC + DN
                  AS(I,J,K) = ASC + DS
                  AT(I,J,K) = ATC + DT
                  AB(I,J,K) = ABC + DB
               END IF
 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

