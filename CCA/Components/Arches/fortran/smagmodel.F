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
C 
C 
c*********************************************************************
c
c
c*********************************************************************

#include <CCA/Components/Arches/fortran/smagmodel_fort.h>
#include <CCA/Components/Arches/fortran/param4.h>



c*********************************************************************
c     Locals
c*********************************************************************
      integer IST, IEND
      integer JST, JEND
      integer KST, KEND

      integer i, j, k
      double precision PMIXL, DMESH
      double precision UNP, USP, UTP, UBP
      double precision VEP, VWP, VTP, VBP
      double precision WEP, WWP, WNP, WSP
      double precision UEP, UWP, VNP, VSP, WTP, WBP
      double precision s11,s22,s33,s12,s13,s23,IsI

c*********************************************************************
c     Get the indices of interest
c*********************************************************************
      IST = domLo(1)
      JST = domLo(2)
      KST = domLo(3)
      IEND = domHi(1)
      JEND = domHi(2)
      KEND = domHi(3)

c*********************************************************************
c     Start
c*********************************************************************
      DO 220 K = KST,KEND
         DO 210 J = JST,JEND
            DO 200 I = IST,IEND
C--------------------------------------------------------------------
C     CALCULATE MIXING OR FILTER LENGTH
C     THIS IS THE SMAGORINSKY MODEL WHEN DOING LES CALCULATIONS (LTIM)
C     Note the discussion in
C     Mason (1994) Q.J.R. Meteorol. Soc., 120, pp. 1-26
C--------------------------------------------------------------------
               DMESH = ((SNS(J)*SEW(I)*STB(K))**(1.0D0/3.0D0))
               PMIXL = CF*MAX(FILTERL,FAC_MESH*DMESH)
C--------------------------------------------------------------------
C     CALCULATE GENERATION OF TURBULENCE
C--------------------------------------------------------------------
               uep = uu(i+1,j,k)
               uwp = uu(i,j,k)
c i,j,k velocity component cancels out when computing derivative,
c so it has been ommited
               UNP = 0.5D0*ucc(I,J+1,K)
               USP = 0.5D0*ucc(I,J-1,K)
               UTP = 0.5D0*ucc(I,J,K+1)
               UBP = 0.5D0*ucc(I,J,K-1)

               vnp = vv(i,j+1,k)
               vsp = vv(i,j,k)
c i,j,k velocity component cancels out when computing derivative,
c so it has been ommited
               VEP = 0.5D0*vcc(I+1,J,K)
               VWP = 0.5D0*vcc(I-1,J,K)
               VTP = 0.5D0*vcc(I,J,K+1)
               VBP = 0.5D0*vcc(I,J,K-1) 

               wtp = ww(i,j,k+1)
               wbp = ww(i,j,k)
c i,j,k velocity component cancels out when computing derivative,
c so it has been ommited
               WEP = 0.5D0*wcc(I+1,J,K)
               WWP = 0.5D0*wcc(I-1,J,K)
               WNP = 0.5D0*wcc(I,J+1,K)
               WSP = 0.5D0*wcc(I,J-1,K)

c              calculate the grid strain rate tensor

               s11 = (uep-uwp)/sew(i)
               s22 = (vnp-vsp)/sns(j)
               s33 = (wtp-wbp)/stb(k)
               s12 = 0.5D0*((unp-usp)/sns(j) + (vep-vwp)/sew(i))
               s13 = 0.5D0*((utp-ubp)/stb(k) + (wep-wwp)/sew(i))
               s23 = 0.5D0*((vtp-vbp)/stb(k) + (wnp-wsp)/sns(j))

c              calculate absolute value of the grid strain rate
   
               IsI = sqrt(2.0D0*(s11**2 + s22**2 + s33**2 +
     &              2.0D0*(s12**2 + s13**2 + s23**2)))

               VIS(I,J,K) = VISCOS + (PMIXL**2)*DEN(I,J,K)*IsI
 200        CONTINUE
 210     CONTINUE
 220  CONTINUE
      RETURN
      END

