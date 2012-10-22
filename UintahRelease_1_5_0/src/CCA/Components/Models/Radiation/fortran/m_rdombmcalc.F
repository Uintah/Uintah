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
#include <CCA/Components/Models/Radiation/fortran/m_rdombmcalc_fort.h>

      integer i,ii,j,jj,k,kk,m

      double precision x,z,relerr,relerr2,big,big2,relerrsq,relerrsq2,
     &rmserrnm,rmserrnm2,small,PI,fluxsume,fluxsumw,fluxsumtb,
     &egerror,areaew,areans,areatb,err,err2,errsq,errsq2,rms_errnm_abs,
     $     rms_errnm2_abs

      DATA PI/3.141593D0/, small/1.0d-10/

      j=10
      k=10
c     above values should be the center of the domain;
c     we are doing a summation along the i-axis
      ii=0
      kk=0
      big=0.0d0
      big2=0.0d0
      relerrsq=0.0d0
      relerrsq2=0.0d0
      errsq = 0.0d0
      errsq2 = 0.0d0
      m=0

c     For energy balance
      
      srcsum = 0.0d0
      fluxsume = 0.0d0
      fluxsumw = 0.0d0
      fluxsumtb = 0.0d0
      egerror = 0.0d0

c      IF (LPROBTWO.OR.LPROBTHREE)GO TO 30 

      IF (LPROBONE)THEN

         z=0.5-zz(kk)
         if (zz(kk).ge.0.5)
     &        z=zz(kk)-0.5

         DO 10 i=idxlo(1),idxhi(1)

            x=0.5-xx(i)
            if (xx(i).ge.0.5)
     &           x=xx(i)-0.5

            srcbm(i) = 55.531*(x**6)-99.436*(x**5)+
     &           74.739*(x**4)-32.32*(x**3)+6.4698*(x**2)-
     &           5.4255*x+3.0755

            qfluxbbm(i) = -20.612*(x**6)+26.792*(x**5)-
     &           13.499*(x**4)+3.6857*(x**3)-0.9482*(x**2)+
     &           0.0104*x+0.188

            if(srcbm(i).ge.big) big=srcbm(i)
            if(qfluxbbm(i).ge.big2) big2=qfluxbbm(i)

 10      continue

         DO 20 i=idxlo(1),idxhi(1)

            if ((j.ge.idxlo(2) .and. j.le.idxhi(2)) .and.
     $           (k.ge.idxlo(3) .and. k.le.idxhi(3))) then

               relerr = (abs(src(i,j,k))-srcbm(i))*100/big
               relerr2 = (abs(qfluxb(i,j,kk))-qfluxbbm(i))*100/big2 
               relerrsq=relerrsq+(relerr**2)
               relerrsq2=relerrsq2+(relerr2**2)

               err = abs(src(i,j,k))-srcbm(i)

               if (abs(src(i,j,k)) .gt. 10*srcbm(i)) then
                  print*, 'found large source at ',i,j,k,'value = ',
     $                 src(i,j,k),srcbm(i)
               end if

               print*,'source at ',i,j,k,' = ',src(i,j,k)
               print*,'flux at ',i,j,kk,' = ',qfluxb(i,j,kk)
               err2 = abs(qfluxb(i,j,kk))-qfluxbbm(i)
               errsq = errsq + err**2
               errsq2 = errsq2 + err2**2

               m=m+1
            endif

 20      continue

         if (m.gt.0) then
            rmserrnm = sqrt(relerrsq/m)
            rmserrnm2 = sqrt(relerrsq2/m)
               
            rms_errnm_abs = sqrt(errsq/m)
            rms_errnm2_abs = sqrt(errsq2/m)

            print*,'outputting rdombmcalc stuff',rmserrnm,rmserrnm2
            print*,'absolute norms = ',rms_errnm_abs,rms_errnm2_abs
         endif

      END IF

 30   continue

      IF (LPROBTWO)THEN

         DO 40 jj=idxlo(2),idxhi(2)
            
c            write(6,*)jj,qfluxw(ii,jj,k)
            
 40      continue

         DO 45 i=idxlo(1),idxhi(1)

            write(6,*)xx(i),src(i,j,k),zz(k)
c            write(6,*)zz(k),qfluxe(i,j,k),xx(i)
c            write(6,*)xx(i),qfluxb(i,j,kk),zz(kk)

 45      continue

      END IF

c     Calculate relative error in energy balance for lprob(one/two)
c     Note: We take advantage of the symmetry in heat fluxes here

      do 60 k=idxlo(3),idxhi(3)
         do 55 j=idxlo(2),idxhi(2)
            do 50 i=idxlo(1),idxhi(1)

               if (pcell(i,j,k).eq.ffield) then

                  volume(i,j,k) = sew(i) * sns(j) * stb(k)
                  srcsum = srcsum - src(i,j,k)*volume(i,j,k)
                  
               endif

               if (pcell(i-1,j,k).ne.ffield) then

                  areaew = sns(j)*stb(k)
                  fluxsumw = fluxsumw + (qfluxw(i,j,k))*areaew 

               endif

               if (pcell(i+1,j,k).ne.ffield) then

                  areaew = sns(j)*stb(k)
                  fluxsume = fluxsume + (qfluxe(i,j,k))*areaew 

               endif

               if (pcell(i,j,k-1).ne.ffield) then

                  areatb = sew(i) * sns(j)
                  fluxsumtb = fluxsumtb + (qfluxb(i,j,k))*areatb

               endif

 50         continue
 55      continue
 60   continue

      egerror = (srcsum - (fluxsume + fluxsumw + 4*fluxsumtb))/srcsum
      write(6,*)"% relative error in energy balance =",egerror*100

      IF (LPROBTHREE)THEN

         DO 65 i=idxlo(1),idxhi(1)

            x=0.32-xx(i)
            if (xx(i).ge.0.32)x=xx(i)-0.32
            if (x.le.small)x=small

            srcbm(i)=(-2/(4*PI*(x**2)))*exp(-2.0*x)
            srcpone(i)=(-1/(4*PI))*((3*(2**2))/x)*exp(-1.732*2*x)
            qfluxbbm(i)=(1/(4*PI*(x**2)))*exp(-2.0*x)

c      write(6,*)x,srcbm(i),-src(i,j,k)
c       write(6,*)x,qfluxbbm(i),volq(i,j,k)

 65      continue

         DO 70 jj=idxlo(2),idxhi(2)
            write(6,*)abs(qfluxw(ii,jj,k))
 70      continue

      END IF
      
      RETURN
      end
