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
#include <CCA/Components/Models/Radiation/fortran/m_radwsgg_fort.h>
C***********************************************************************
C	SUBROUTINE RADWSGG - calculates spectral gas and soot absorption  
C	coefficients for cartesian or cylindrical geometries.
C	(particle radiation properties are calculated in GLACIER)
C***********************************************************************
C      IMPLICIT NONE 

      double precision PI,PI1,SIG,twall,temp,a,fracn,fracnsum,fracntot,
     &scatcoef,phfunc,twalltwo,abskgtemp

      double precision b(4),c(3,4,4),d(4,4)

      Integer I,J,K
      PI =dacos(-1.0D0)
      DATA SIG/5.67E-08/

      DATA c(1,1,1)/0.59324d0/,c(1,1,2)/-0.61741d-03/,
     &c(1,1,3)/0.29248d-06/,c(1,1,4)/-0.45823d-10/,
     &c(1,2,1)/0.35739d-03/,c(1,2,2)/0.22122d-06/,
     &c(1,2,3)/-0.26380d-09/,c(1,2,4)/0.45951d-13/,
     &c(1,3,1)/-0.71313d-06/,c(1,3,2)/0.46181d-09/,
     &c(1,3,3)/-0.70858d-13/,c(1,3,4)/0.38038d-17/,
     &c(1,4,1)/0.17806d-09/,c(1,4,2)/-0.11654d-12/,
     &c(1,4,3)/0.19939d-16/,c(1,4,4)/-0.13486d-20/,
     &c(2,1,1)/-0.35664d-01/,c(2,1,2)/0.21502d-03/,
     &c(2,1,3)/-0.13648d-06/,c(2,1,4)/0.24284d-10/,
     &c(2,2,1)/0.51605d-03/,c(2,2,2)/-0.70037d-06/,
     &c(2,2,3)/0.38680d-09/,c(2,2,4)/0.70429d-13/,
     &c(2,3,1)/0.12245d-06/,c(2,3,2)/0.99434d-10/,
     &c(2,3,3)/-0.15598d-12/,c(2,3,4)/0.37664d-16/,
     &c(2,4,1)/-0.57563d-10/,c(2,4,2)/-0.10109d-13/,
     &c(2,4,3)/0.35273d-16/,c(2,4,4)/-0.89872d-20/,
     &c(3,1,1)/0.12951d-00/,c(3,1,2)/0.54520d-04/,
     &c(3,1,3)/-0.80049d-07/,c(3,1,4)/0.17813d-10/,
     &c(3,2,1)/0.15210d-03/,c(3,2,2)/-0.37750d-06/,
     &c(3,2,3)/0.21019d-09/,c(3,2,4)/-0.36011d-13/,
     &c(3,3,1)/-0.13165d-06/,c(3,3,2)/0.20719d-09/,
     &c(3,3,3)/-0.96720d-13/,c(3,3,4)/0.14807d-16/,
     &c(3,4,1)/0.26872d-10/,c(3,4,2)/-0.34803d-13/,
     &c(3,4,3)/0.14336d-16/,c(3,4,4)/-0.19754d-20/

      PI1 = 1.0d0 / PI

      TWALL = 293.0d0
      TWALLTWO = 293.0d0

      if (bands.eq.1) then

         a = 0.4201d0
         b(1) = 6.508d0
         b(2) = -5.551d0
         b(3) = 3.029d0
         b(4) = -5.353d0

      else if (bands.eq.2) then
 
         a = 6.516d0
         b(1) = -0.2504d0
         b(2) = 6.112d0
         b(3) = -3.882d0
         b(4) = 6.528d0

      else if (bands.eq.3) then 

         a = 131.9d0 
         b(1) = 2.718d0
         b(2) = -3.118d0
         b(3) = 1.221d0
         b(4) = -1.612d0

      else if (bands.eq.4) then 

         a = 0.0d0 
         b(1) = 8.9756d0
         b(2) = -2.557d0
         b(3) = 0.368d0
         b(4) = -0.437d0
      end if

      DO 80 K=idxlo(3)-1,idxhi(3)+1
        DO 70 J=idxlo(2)-1,idxhi(2)+1
          DO 60 I=idxlo(1)-1,idxhi(1)+1

             if ((i.eq.idxlo(1)-1 .and. j.eq.idxlo(2)-1) .or.
     $            (i.eq.idxlo(1)-1 .and. k.eq.idxlo(3)-1) .or.
     $            (i.eq.idxlo(1)-1 .and. k.eq.idxhi(3)+1) .or.
     $            (i.eq.idxlo(1)-1 .and. j.eq.idxhi(2)+1) .or.
c
     $            (i.eq.idxhi(1)+1 .and. j.eq.idxlo(2)-1) .or.
     $            (i.eq.idxhi(1)+1 .and. k.eq.idxlo(3)-1) .or.
     $            (i.eq.idxhi(1)+1 .and. j.eq.idxhi(2)+1) .or.
     $            (i.eq.idxhi(1)+1 .and. k.eq.idxhi(3)+1) .or.
c
     $            (j.eq.idxlo(2)-1 .and. k.eq.idxlo(3)-1) .or.
     $            (j.eq.idxlo(2)-1 .and. k.eq.idxhi(3)+1) .or.
     $            (j.eq.idxhi(2)+1 .and. k.eq.idxlo(3)-1) .or.
     $            (j.eq.idxhi(2)+1 .and. k.eq.idxhi(3)+1)
c
     $            ) goto 666

             IF (PCELL(I,J,K).EQ.FFIELD) THEN

                temp = TG(I,J,K)
                ABSKG(I,J,K) = a*(CO2(I,J,K)+H2O(I,J,K))+
     &               402.834d0*SFV(I,J,K)*TG(I,J,K)
                
                abskgtemp = abskg(i,j,k)
                if(abskgtemp.le.1.0d-06)abskgtemp = 1.0d-06

c     Anisotropic scattering properties for the interior nodes
c     (for spherical harmonics only)

                scatcoef = 0.0d0
                phfunc   = 0.0d0
                shgamma(i,j,k) = 3.0d0*(abskgtemp + scatcoef) - 
     &               phfunc*scatcoef
                shgamma(i,j,k) = 1.0d0/shgamma(i,j,k)
                
                fracn = (b(1)*1E-01)+
     &               (b(2)*1E-04)*temp +
     &               (b(3)*1E-07)*(temp**2)+
     &               (b(4)*1E-011)*(temp**3)

                if (bands.eq.4) then
                   fracn = 1.0d0 - fracn
                end if

                ESRCG(I,J,K) = abs(fracn) * SIG * PI1 * 
     &               ABSKG(I,J,K) * TG(I,J,K)**4

             ENDIF
             
 666         continue

 60       CONTINUE
 70    CONTINUE
 80   CONTINUE

c     Emissivity curve-fit stuff for wall bc's

      fraction(bands) = (b(1)*1E-01)+
     &     (b(2)*1E-04)*twall +
     &     (b(3)*1E-07)*(twall**2)+
     &     (b(4)*1E-011)*(twall**3)
      
      if (bands.eq.4) then
         fraction(bands) = 1.0d0 - fraction(bands)
      end if

c     Emissivity curve-fit stuff for the second wall bc's

      fractiontwo(bands) = (b(1)*1E-01)+
     &     (b(2)*1E-04)*twalltwo +
     &     (b(3)*1E-07)*(twalltwo**2)+
     &     (b(4)*1E-011)*(twalltwo**3)

      if (bands.eq.4) then
         fractiontwo(bands) = 1.0d0 - fractiontwo(bands)
      end if

c     Absorptivity curve-fit stuff

c      fraction(bands)=0.0d0

c	do 100 j=1,4
c      fracnsum = 0.0d0
c	   do 90 k=1,4

c	      fracnsum=fracnsum+c(bands,j,k)*(twall**(k-1))

c 90	   continue

c      fraction(bands)=fraction(bands)+
c     &fracnsum*(1467.0d0**(j-1))

c 100	continue


c	if (bands.eq.4) then

c      fracntot = 0.0d0

c	   do 130 i=1,3

c	    fraction(i) = 0.0d0

c	      do 120 j=1,4
c              fracnsum = 0.0d0
c		 do 110 k=1,4

c	      fracnsum=fracnsum+c(i,j,k)*(twall**(k-1))

c110	         continue

c	      fraction(i)=fraction(i)+fracnsum*(1467.0d0**(j-1))

c120	      continue

c	      fracntot = fracntot + fraction(i)

c130	   continue
           
c	   fraction(bands) = 1.0d0 - fracntot

c	end if

c	write(6,*)bands,fraction(bands)

	RETURN
	END






