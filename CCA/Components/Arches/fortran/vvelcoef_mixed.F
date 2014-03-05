C
C The MIT License
C
C Copyright (c) 1997-2012 The University of Utah
C
C Permission is hereby granted, free of charge, to any person obtaining a copy
C of this software and associated documentation files (the "Software"), to
C deal in the Software without restriction, including without limitation the
C rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/vvelcoef_mixed_fort.h>
C---------------------------------------------------------------------
C     THREE-DIMENSIONAL VERSION
C     THIS SUBROUTINE WILL compute matrix coefficients for vvelocity
C---------------------------------------------------------------------

c*********************************************************************
c     Local Variables :
c*********************************************************************
      integer ISTV, JSTV, KSTV, IENDV, JENDV, KENDV
      integer i, j, k
      integer flag

      double precision AREANS, AREAEW, AREATB, VOL
      double precision CE, CW, CN, CS, CT, CB
      double precision GE, GP, GW, GN, GSE, GS, GSW, GT, GST, GSB, GB
      double precision GSP, GNP, GUP
      double precision GWP
      double precision CPE, CPW, GVP, CPN, CPS, CPT, CPB
      double precision VISE, VISW, VISN, VISS, VIST, VISB
      double precision DE, DW, DN, DS, DT, DB
      double precision TEW, TNS, TTB
      double precision COEFE, COEFW, COEFN, COEFS, COEFT, COEFB
      double precision AEC, AWC, ANC, ASC, ATC, ABC
      double precision Relimit_m1

c*********************************************************************
c     Start :
c*********************************************************************
      ISTV = idxLoV(1)
      JSTV = idxLoV(2)
      KSTV = idxLoV(3)
      IENDV = idxHiV(1)
      JENDV = idxHiV(2)
      KENDV = idxHiV(3)

      AREANS = dx*dz
      AREAEW = dy*dz
      AREATB = dx*dy
      VOL = dx*dy*dz

      Relimit_m1 = Relimit**-1

      DO 220 K = KSTV,KENDV
         DO 210 J = JSTV,JENDV
            DO 200 I = ISTV,IENDV

              flag = 0

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
               GT = 0.5D0*(DEN(I,J,K+1)+DEN(I,J,K))*WW(I,J,K+1)
               GST = 0.5D0*(DEN(I,J-1,K+1)+DEN(I,J-1,K))*WW(I,J-1,K+1)
               GB = 0.5D0*(DEN(I,J,K)+DEN(I,J,K-1))*WW(I,J,K)
               GSB = 0.5D0*(DEN(I,J-1,K)+DEN(I,J-1,K-1))*WW(I,J-1,K)

               CE = (0.5D0*GE+0.5D0*GSE)*AREAEW
               CW = (0.5D0*GW+0.5D0*GSW)*AREAEW
               CT = (0.5D0*GT+0.5D0*GST)*AREATB
               CB = (0.5D0*GB+0.5D0*GSB)*AREATB 
               CN = 0.5D0*(GN+GP)*AREANS
               CS = 0.5D0*(GP+GS)*AREANS

               CESAV(I,J,K) = CE
               CWSAV(I,J,K) = CW
               CNSAV(I,J,K) = CN
               CSSAV(I,J,K) = CS
               CTSAV(I,J,K) = CT
               CBSAV(I,J,K) = CB

               GUP = 0.25*(GE+GSE+GW+GSW)
               CPE = GUP*VOL/DX
               CPW = GUP*VOL/DX
               CPN = GP*VOL/DY
               CPS = GP*VOL/DY
               GWP = 0.25*(GT+GST+GB+GSB)
               CPT = GWP*VOL/DZ
               CPB = GWP*VOL/DZ

               COEFE = DE - Relimit_m1*ABS(CE)
               COEFW = DW - Relimit_m1*ABS(CW)
               COEFN = DN - Relimit_m1*ABS(CN)
               COEFS = DS - Relimit_m1*ABS(CS)
               COEFT = DT - Relimit_m1*ABS(CT)
               COEFB = DB - Relimit_m1*ABS(CB)

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

C-----------------------------------------------------------------------
C     CALCULATE DIFFUSION COEFFICIENTS
C-----------------------------------------------------------------------
               VISE = 0.25D0*(VIS(I+1,J,K)+VIS(I+1,J-1,K))+
     +                0.25D0*(VIS(I,J,K)+ VIS(I,J-1,K))
               VISW = 0.25D0*(VIS(I-1,J,K)+VIS(I-1,J-1,K))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J-1,K))
               VISN = VIS(I,J,K)
               VISS = VIS(I,J-1,K)
               VIST = 0.25D0*(VIS(I,J,K+1)+VIS(I,J-1,K+1))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J-1,K))
               VISB = 0.25D0*(VIS(I,J,K-1)+VIS(I,J-1,K-1))+
     +                0.25D0*(VIS(I,J,K)+VIS(I,J-1,K))

               DE = VISE*AREAEW/DX
               DW = VISW*AREAEW/DX
               DN = VISN*AREANS/DY
               DS = VISS*AREANS/DY
               DT = VIST*AREATB/DZ
               DB = VISB*AREATB/DZ

C-----------------------------------------------------------------------
C     CONVECTION DIFFERENCING SCHEME SELECTION
C-----------------------------------------------------------------------
               IF ( celltype(I+1,J,K) .eq. WALL .or. 
     $             celltype(I+1,J,K) .eq. INTR ) THEN
                 flag = 1 
               ELSEIF ( celltype(I+1,J-1,K) .eq. WALL .or. 
     $             celltype(I+1,J-1,K) .eq. INTR ) THEN 
                 flag = 1 
               ENDIF 

               if ( celltype(i-1,j,k) .eq. WALL .or. 
     $              celltype(i-1,j,k) .eq. INTR ) then 
                 flag = 1 
               elseif ( celltype(i-1,j-1,k) .eq. WALL .or. 
     $              celltype(i-1,j-1,k) .eq. INTR) then 
                 flag = 1 
               endif 

               if ( celltype(i,j,k+1) .eq. WALL .or. 
     $              celltype(i,j,k+1) .eq. INTR ) then 
                 flag = 1 
               elseif ( celltype(i,j-1,k+1) .eq. WALL .or.
     $              celltype(i,j-1,k+1) .eq. INTR ) then 
                 flag = 1 
               endif 

               if ( celltype(i,j,k-1) .eq. WALL .or.
     $              celltype(i,j,k-1) .eq. INTR ) then 
                 flag = 1 
               elseif ( celltype(i,j-1,k-1) .eq. WALL .or. 
     $              celltype(i,j-1,k-1) .eq. INTR ) then 
                 flag = 1 
               endif 

               if ( flag .eq. 1 ) then 
                 AE(I,J,K) = AEC + DE
                 AW(I,J,K) = AWC + DW
                 AN(I,J,K) = ANC + DN
                 AS(I,J,K) = ASC + DS
                 AT(I,J,K) = ATC + DT
                 AB(I,J,K) = ABC + DB
               else
                 AE(I,J,K) = DE-CE*0.5D0
                 AW(I,J,K) = DW+CW*0.5D0
                 AN(I,J,K) = DN-CN*0.5D0
                 AS(I,J,K) = DS+CS*0.5D0
                 AT(I,J,K) = DT-CT*0.5D0
                 AB(I,J,K) = DB+CB*0.5D0
               endif


 200        CONTINUE
 210     CONTINUE
 220  CONTINUE

      RETURN
      END

