c
c

#include <Packages/Uintah/CCA/Components/Arches/fortran/scalcoef_fort.h>
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
C	prinv	Inverse of the "Prandtl number" for this scalar
C		(e.g., PRF, PRG, etc.) (i/p)
C	crn	Average Cell Reynolds number (o/p)
C	vol	Array: Volume of P-Cell (o/p)
C	rv	Radius of V-velocity node point (i/p)
C	ra	Radius of P-cell center (i/p)
C	sew	Width of P-cell in x-direction (i/p)
C	sns	Width of P-cell in y-direction (i/p)
C	stb	Width of P-cell in z-direction (i/p)
C	rone	Dummy radius array, = 1.0 (i/p)
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
C	enfac,sfac	
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
      double precision prinv
      double precision crn
      logical*1 l2up, lsmpt, lcend, lcentral, leno
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
      double precision crn0
      double precision prandtlno, prno

      integer i, j, k
      integer IST, JST, KST
      integer IEND, JEND, KEND
*-----------------------------------------------------------------------

      prinv = 0.01D0
      crn = zero
      l2up = .false.
      lcend = .false.
      lsmpt = .false.
      lcentral = .false.
      leno = .false.
      if (conv_scheme.eq.0) then
        l2up = .true.
      else
        leno = .true.
      endif
      vol = zero
      prno = 0.4D0

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
               if (lcentral) then
                  CE = 0.5*(CEO + CWO)
                  CW = 0.5*(CWO + CEO)
                  CN = 0.5*(CNO + CSO)
                  CS = 0.5*(CNO + CSO)
                  CT = 0.5*(CTO + CBO)
                  CB = 0.5*(CTO + CBO)
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
               GUP = DEN(I,J,K)*(EFAC(I)*UU(I+1,J,K)+WFAC(I)*UU(I,J,K))
               CPE = GUP*VOL/DXEP(I)
               CPW = GUP*VOL/DXPW(I)
               GVP = DEN(I,J,K)*(ENFAC(J)*VV(I,J+1,K)+SFAC(J)*VV(I,J,K))
               CPN = GVP*VOL/DYNP(J)
               CPS = GVP*VOL/DYPS(J)
               GWP = DEN(I,J,K)*(TFAC(K)*WW(I,J,K+1)+BFAC(K)*WW(I,J,K))
               CPT = GWP*VOL/DZTP(K)
               CPB = GWP*VOL/DZPB(K)
C--------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C--------------------------------------------------------------------
               if (i .lt. 10) then
                  prandtlno = prno
               else
                  prandtlno = prno
               endif
               GAME = 0.5D0*(VIS(I,J,K)+VIS(I+1,J,K))/prandtlno
               GAMW = 0.5D0*(VIS(I,J,K)+VIS(I-1,J,K))/prandtlno
               GAMN = 0.5D0*(VIS(I,J,K)+VIS(I,J+1,K))/prandtlno
               GAMS = 0.5D0*(VIS(I,J,K)+VIS(I,J-1,K))/prandtlno
               GAMT = 0.5D0*(VIS(I,J,K)+VIS(I,J,K+1))/prandtlno
               GAMB = 0.5D0*(VIS(I,J,K)+VIS(I,J,K-1))/prandtlno
               DEO = GAME*AREAEW/DXEP(I)
               DWO = GAMW*AREAEW/DXPW(I)
               DNO = GAMN*AREAN/DYNP(J)
               DSO = GAMS*AREAS/DYPS(J)
               DTO = GAMT*AREATB/DZTP(K)
               DBO = GAMB*AREATB/DZPB(K)
C-----------------------------------------------------------------------
C     NEW DIFFUSION TERMS
C-----------------------------------------------------------------------
               DE = CEE(I)*DEO + CWE(I)*DWO*DXPW(I)/DXEP(I)
               DW = CWW(I)*DWO - CWE(I)*DEO*DXEP(I)/DXPW(I)   
               DN = CNN(J)*DNO + CSN(J)*DSO*DYPS(J)/DYNP(J)
               DS = CSS(J)*DSO - CSN(J)*DNO*DYNP(J)/DYPS(J)
               DT = CTT(K)*DTO + CBT(K)*DBO*DZPB(K)/DZTP(K)
               DB = CBB(K)*DBO - CBT(K)*DTO*DZTP(K)/DZPB(K)
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
c === double precision stuff for aix ==               
               AEC = -0.5D0*CE*TEW + dMAX1(0.0D0,-CPE)*(1.0D0-TEW)
               AWC = 0.5D0*CW*TEW + dMAX1(0.0D0,CPW)*(1.0D0-TEW)
               ANC = -0.5D0*CN*TNS + dMAX1(0.0D0,-CPN)*(1.0D0-TNS)
               ASC = 0.5D0*CS*TNS + dMAX1(0.0D0,CPS)*(1.0D0-TNS)
               ATC = -0.5D0*CT*TTB + dMAX1(0.0D0,-CPT)*(1.0D0-TTB)
               ABC = 0.5D0*CB*TTB + dMAX1(0.0D0,CPB)*(1.0D0-TTB)
c === original code ===
c               AEC = -0.5*CE*TEW + MAX(0.0,-CPE)*(1.0-TEW)
c               AWC = 0.5*CW*TEW + MAX(0.0,CPW)*(1.0-TEW)
c               ANC = -0.5*CN*TNS + MAX(0.0,-CPN)*(1.0-TNS)
c               ASC = 0.5*CS*TNS + MAX(0.0,CPS)*(1.0-TNS)
c               ATC = -0.5*CT*TTB + MAX(0.0,-CPT)*(1.0-TTB)
c               ABC = 0.5*CB*TTB + MAX(0.0,CPB)*(1.0-TTB)
C----------------------------------------------------------------------
C      PROVISIONS FOR VARIOUS DIFFERENCING SCHEMES
C----------------------------------------------------------------------
               IF (LSMPT) THEN
                  AE(I,J,K) = DE
                  AW(I,J,K) = DW
                  AN(I,J,K) = DN
                  AS(I,J,K) = DS
                  AT(I,J,K) = DT
                  AB(I,J,K) = DB
c                  AC(I,J,K) = AEC+AWC+ANC+ASC+ATC+ABC
               ELSE IF (LENO) THEN
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
c                  AC(I,J,K) = 0.0
               ELSE IF (LCENTRAL) THEN
                  AE(I,J,K) = DE-CE*0.5D0
                  AW(I,J,K) = DW+CW*0.5D0
                  AN(I,J,K) = DN-CN*0.5D0
                  AS(I,J,K) = DS+CS*0.5D0
                  AT(I,J,K) = DT-CT*0.5D0
                  AB(I,J,K) = DB+CB*0.5D0
c                  AC(I,J,K) = 0.0D0
               ELSE IF (L2UP) THEN
c === double precision stuff for aix                  
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
C === original code ===
c                  AE(I,J,K) = DE*(1.0-MAX(0.0,SIGN(1.0,COEFE)))
c     &                 + MAX(0.0,COEFE) + MAX(0.0,-CE)
c                  AW(I,J,K) = DW*(1.0-MAX(0.0,SIGN(1.0,COEFW)))
c     &                 + MAX(0.0,COEFW) + MAX(0.0,CW)
c                  AN(I,J,K) = DN*(1.0-MAX(0.0,SIGN(1.0,COEFN)))
c     &                 + MAX(0.0,COEFN) + MAX(0.0,-CN)
c                  AS(I,J,K) = DS*(1.0-MAX(0.0,SIGN(1.0,COEFS)))
c     &                 + MAX(0.0,COEFS) + MAX(0.0,CS)
c                  AT(I,J,K) = DT*(1.0-MAX(0.0,SIGN(1.0,COEFT)))
c     &                 + MAX(0.0,COEFT) + MAX(0.0,-CT)
c                  AB(I,J,K) = DB*(1.0-MAX(0.0,SIGN(1.0,COEFB)))
c     &                 + MAX(0.0,COEFB) + MAX(0.0,CB)
c                  AC(I,J,K) = 0.0
               ELSE
                  AE(I,J,K) = AEC + DE
                  AW(I,J,K) = AWC + DW
                  AN(I,J,K) = ANC + DN
                  AS(I,J,K) = ASC + DS
                  AT(I,J,K) = ATC + DT
                  AB(I,J,K) = ABC + DB
c                  AC(I,J,K) = 0.0
               END IF

 203        continue
 210     continue
 220  continue

      return
      end

c
c Revision 1.6  2000/10/10 19:31:03  rawat
c added scalarsolver
c
c Revision 1.5  2000/09/28 22:16:47  bigler
c Changed certain functions to explicity be double precision functions.
c
c Revision 1.4  2000/08/19 16:36:40  rawat
c fixed some bugs in scalarcoef calculations
c
c Revision 1.3  2000/08/09 20:19:27  rawat
c modified scalcoef.F
c
c Revision 1.2  2000/08/08 23:34:20  rawat
c fixed some bugs in profv.F and Properties.cc
c
c Revision 1.1  2000/07/14 05:23:52  bbanerje
c Added scalcoef.F and updated related stuff in C++. scalcoef ==> coefs.f
c in Kumar's code.
c
c
