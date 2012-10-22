C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
C sell copies of the Software, and to permit persons to whom the Software is
C furnished to do so, subject to the following conditions:
C
C The above copyright notice and this permission notice shall be included in
C all copies or substantial portions of the Software.
C
C THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
C IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
C AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
C IN THE SOFTWARE.
C
C 
C 
c
c

#include <CCA/Components/Arches/fortran/scalcoef_fort.h>
#include "param4.h"
C-----------------------------------------------------------------------
C     WRITTEN BY SESHADRI KUMAR, OCTOBER 14, 1996
C
C     COEFS calculates the seven diagonals of the seven-point finite
C     difference equation for any scalar equation (e.g., F,G, TKE,
C     TED, ETA, etc.) 
C
C     PASSED VARIABLES:
C
C	den	Density field (i/p)
C	vis	Viscosity field (i/p)
C	u	X-directed velocity (i/p)
C	v	Y- or r-directed velocity (i/p)
C	w 	z- or theta-directed velocity (i/p)
C	ae,aw,an,as,at,ab,ap
C		Coefficients of the seven-diagonal finite difference
C		equation (o/p)
C	cesav,cwsav,cnsav,cssav,ctsav,cbsaV
C		Convection fluxes on the east,west,north,south,top,
C		and bottom faces of the P-cell (o/p)
C	vol	Array: Volume of P-Cell (o/p)
C	sew	Width of P-cell in x-direction (i/p)
C	sns	Width of P-cell in y-direction (i/p)
C	stb	Width of P-cell in z-direction (i/p)
C	cee,cwe,cww
C		Interpolation factors for interpolating from
C		E- and W-faces of regular P-cell to artificial
C		cell for accurate second order finite differences (i/p)
C	cnn,csn,css
C		Interpolation factors for interpolating from
C		N- and S-faces of regular P-cell to artificial
C		cell for accurate second order finite differences (i/p)
C	ctt,cbt,cbb
C		Interpolation factors for interpolating from
C		T- and B-faces of regular P-cell to artificial
C		cell for accurate second order finite differences (i/p)
C	efac,wfac	
C		Interpolation factors for calculating value at
C		P-node from values at its east and west cell faces
C		respectively (i/p)
C	nfac,sfac	
C		Interpolation factors for calculating value at
C		P-node from values at its north and south cell faces
C		respectively (i/p)
C	tfac,bfac	
C		Interpolation factors for calculating value at
C		P-node from values at its top and bottom cell faces
C		respectively (i/p)
C	dxpw	Distance between successive P-nodes in the x-direction
C		(see cellg.f) (i/p)
C	dyps	Distance between successive P-nodes in the y-direction
C		(i/p)
C	dzpb	Distance between successive P-nodes in the z-direction
C		(i/p)
C	ist,jst,kst
C		Starting indices for the calculation of the coefficients
C		(i/p)
C	iend,jend,kend
C		Ending indices for calculation (i/p)
C	l2up	Logical, =T implies that the 2nd upwind differencing
C		scheme of Roache is used
C
C-----------------------------------------------------------------------

*-----------------------------------------------------------------------
*     local variables
*
      logical*1 l2up, lcend, leno
      double precision vol
      double precision arean, areas, areaew, areatb
      double precision dene, denw, denn, dens, dent, denb
      double precision ceo, cwo, cno, cso, cto, cbo
      double precision ce, cw, cn, cs, ct, cb
      double precision gup, cpe, cpw, gvp, cpn, cps, gwp, cpt, cpb
      double precision game, gamw, gamn, gams, gamt, gamb
      double precision deo, dwo, dno, dso, dto, dbo
      double precision de, dw, dn, ds, dt, db
      double precision coefe, coefw, coefn, coefs, coeft, coefb
      double precision tew, tns, ttb
      double precision aec, awc, anc, asc, atc, abc

      integer i, j, k
      integer IST, JST, KST
      integer IEND, JEND, KEND
*-----------------------------------------------------------------------

      l2up = .false.
      lcend = .false.
      leno = .false.
      if (conv_scheme.eq.0) then
        l2up = .true.
      else
        leno = .true.
      endif
      vol = zero

      IST = idxLo(1)
      JST = idxLo(2)
      KST = idxLo(3)
      IEND = idxHi(1)
      JEND = idxHi(2)
      KEND = idxHi(3)

      do 220 k = KST, KEND
         do 210 j = JST, JEND
            do 203 i = IST, IEND

c-----compute areas and volume------------------------------------------

               AREAN = SEW(I)*STB(K)
               AREAS = SEW(I)*STB(K)
               AREAEW = SNS(J)*STB(K)
               AREATB = SEW(I)*SNS(J)
               vol = sns(j)*sew(i)*stb(k)
C-----------------------------------------------------------------------
C     CALCULATE CONVECTION COEFFICIENTS
C-----------------------------------------------------------------------
               DENE = 0.5D0*(DEN(I,J,K)+DEN(I+1,J,K))
               DENW = 0.5D0*(DEN(I,J,K)+DEN(I-1,J,K))
               DENN = 0.5D0*(DEN(I,J,K)+DEN(I,J+1,K))
               DENS = 0.5D0*(DEN(I,J,K)+DEN(I,J-1,K))
               DENT = 0.5D0*(DEN(I,J,K)+DEN(I,J,K+1))
               DENB = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))
               CEO = DENE*UU(I+1,J,K)*AREAEW
               CWO = DENW*UU(I,J,K)*AREAEW
               CNO = DENN*VV(I,J+1,K)*AREAN
               CSO = DENS*VV(I,J,K)*AREAS
               CTO = DENT*WW(I,J,K+1)*AREATB
               CBO = DENB*WW(I,J,K)*AREATB
C-----------------------------------------------------------------------
C      NEW DIFFERENCING STUFF
C-----------------------------------------------------------------------
               CE = CEE(I)*CEO + CWE(I)*CWO
               CW = CWW(I)*CWO - CWE(I)*CEO   
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

C-----------------------------------------------------------------------
C      UPWIND DIFFERENCING STUFF
C-----------------------------------------------------------------------
               GUP = DEN(I,J,K)*(EFAC(I)*UU(I+1,J,K)+WFAC(I)*UU(I,J,K))
               CPE = GUP*VOL/DXEP(I)
               CPW = GUP*VOL/DXPW(I)
               GVP = DEN(I,J,K)*(NFAC(J)*VV(I,J+1,K)+SFAC(J)*VV(I,J,K))
               CPN = GVP*VOL/DYNP(J)
               CPS = GVP*VOL/DYPS(J)
               GWP = DEN(I,J,K)*(TFAC(K)*WW(I,J,K+1)+BFAC(K)*WW(I,J,K))
               CPT = GWP*VOL/DZTP(K)
               CPB = GWP*VOL/DZPB(K)
C--------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C--------------------------------------------------------------------
c               GAME = 0.5D0*(VIS(I,J,K)+VIS(I+1,J,K))/prandtlno
c               GAMW = 0.5D0*(VIS(I,J,K)+VIS(I-1,J,K))/prandtlno
c               GAMN = 0.5D0*(VIS(I,J,K)+VIS(I,J+1,K))/prandtlno
c               GAMS = 0.5D0*(VIS(I,J,K)+VIS(I,J-1,K))/prandtlno
c               GAMT = 0.5D0*(VIS(I,J,K)+VIS(I,J,K+1))/prandtlno
c               GAMB = 0.5D0*(VIS(I,J,K)+VIS(I,J,K-1))/prandtlno
               GAME = (FAC1EW(I)*VIS(I+1,J,K)+
     +                 FAC2EW(I)*VIS(I+E_shift(I),J,K))/prandtlno
               GAMW = (FAC3EW(I)*VIS(I-1,J,K)+
     +                 FAC4EW(I)*VIS(I+W_shift(I),J,K))/prandtlno
               GAMN = (FAC1NS(J)*VIS(I,J+1,K)+
     +                 FAC2NS(J)*VIS(I,J+N_shift(J),K))/prandtlno
               GAMS = (FAC3NS(J)*VIS(I,J-1,K)+
     +                 FAC4NS(J)*VIS(I,J+S_shift(J),K))/prandtlno
               GAMT = (FAC1TB(K)*VIS(I,J,K+1)+
     +                 FAC2TB(K)*VIS(I,J,K+T_shift(K)))/prandtlno
               GAMB = (FAC3TB(K)*VIS(I,J,K-1)+
     +                 FAC4TB(K)*VIS(I,J,K+B_shift(K)))/prandtlno
               DEO = GAME*AREAEW/DXEP(I)
               DWO = GAMW*AREAEW/DXPW(I)
               DNO = GAMN*AREAN/DYNP(J)
               DSO = GAMS*AREAS/DYPS(J)
               DTO = GAMT*AREATB/DZTP(K)
               DBO = GAMB*AREATB/DZPB(K)
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
c               DT = CTT(K)*DTO + CBT(K)*DBO*DZPB(K)/DZTP(K)
c               DB = CBB(K)*DBO - CBT(K)*DTO*DZTP(K)/DZPB(K)
               DE = DEO
               DW = DWO   
               DN = DNO
               DS = DSO
               DT = DTO
               DB = DBO
               if (de.lt.0.0D0) write(6,*) "DE CC is negative"
               if (dw.lt.0.0D0) write(6,*) "DW CC is negative"
               if (dn.lt.0.0D0) write(6,*) "DN CC is negative"
               if (ds.lt.0.0D0) write(6,*) "DS CC is negative"
               if (dt.lt.0.0D0) write(6,*) "DT CC is negative"
               if (db.lt.0.0D0) write(6,*) "DB CC is negative"
               if ((de.lt.0.0D0).or.(dw.lt.0.0D0).or.(dn.lt.0.0D0).or.
     .             (ds.lt.0.0D0).or.(dt.lt.0.0D0).or.(db.lt.0.0D0)) then
                 write(6,*) "Location indexes: ",i,j,k
                 write(6,*) "Possibly stretching is two high."
                 call exit(0)
               end if
               DECOEF(I,J,K) = DE
               DWCOEF(I,J,K) = DW
               DNCOEF(I,J,K) = DN
               DSCOEF(I,J,K) = DS
               DTCOEF(I,J,K) = DT
               DBCOEF(I,J,K) = DB
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

               AEC = -0.5D0*CE*TEW + dMAX1(0.0D0,-CPE)*(1.0D0-TEW)
               AWC = 0.5D0*CW*TEW + dMAX1(0.0D0,CPW)*(1.0D0-TEW)
               ANC = -0.5D0*CN*TNS + dMAX1(0.0D0,-CPN)*(1.0D0-TNS)
               ASC = 0.5D0*CS*TNS + dMAX1(0.0D0,CPS)*(1.0D0-TNS)
               ATC = -0.5D0*CT*TTB + dMAX1(0.0D0,-CPT)*(1.0D0-TTB)
               ABC = 0.5D0*CB*TTB + dMAX1(0.0D0,CPB)*(1.0D0-TTB)
C----------------------------------------------------------------------
C      PROVISIONS FOR VARIOUS DIFFERENCING SCHEMES
C----------------------------------------------------------------------
               IF (LENO) THEN
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
               ELSE IF (LCEND) THEN
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

 203        continue
 210     continue
 220  continue

      return
      end

