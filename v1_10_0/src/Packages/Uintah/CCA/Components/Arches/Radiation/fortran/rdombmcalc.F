#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdombmcalc_fort.h>

      integer i,ii,j,jj,k,kk,m

      double precision x,z,relerr,relerr2,big,big2,relerrsq,relerrsq2,
     &rmserrnm,rmserrnm2,small,PI

      DATA PI/3.141593D0/, small/1.0d-10/

      j=12
      k=12
      ii=0
      kk=0
      big=0.0d0
      big2=0.0d0
      relerrsq=0.0d0
      relerrsq2=0.0d0
      m=0

      IF (LPROBTWO.OR.LPROBTHREE)GO TO 30 

            z=0.5-zz(kk)
            if (zz(kk).ge.0.5)
     &      z=zz(kk)-0.5

      DO 10 i=idxlo(1),idxhi(1)

            x=0.5-xx(i)
            if (xx(i).ge.0.5)
     &      x=xx(i)-0.5

            srcbm(i) = 55.531*(x**6)-99.436*(x**5)+
     &      74.739*(x**4)-32.32*(x**3)+6.4698*(x**2)-
     &      5.4255*x+3.0755

            qfluxbbm(i) = -20.612*(x**6)+26.792*(x**5)-
     &      13.499*(x**4)+3.6857*(x**3)-0.9482*(x**2)+
     &      0.0104*x+0.188

      if(srcbm(i).ge.big)
     &big=srcbm(i)

      if(qfluxbbm(i).ge.big2)
     &big2=qfluxbbm(i)

10    continue

      DO 20 i=idxlo(1),idxhi(1)

      relerr = (abs(src(i,j,k))-srcbm(i))*100/big

      relerr2 = (abs(qfluxb(i,j,kk))-qfluxbbm(i))*100/big2 

      relerrsq=relerrsq+(relerr**2)
      relerrsq2=relerrsq2+(relerr2**2)

      m=m+1

20    continue

      rmserrnm = sqrt(relerrsq/m)
      rmserrnm2 = sqrt(relerrsq2/m)

      write(6,*)rmserrnm,rmserrnm2

30    continue

      IF (LPROBTWO)THEN

c      DO 40 jj=idxlo(2),idxhi(2)

c      write(6,*)jj,qfluxw(ii,jj,k)

c40    continue

      DO 45 i=idxlo(1),idxhi(1)

      write(6,*)xx(i),src(i,j,k)

45    continue

      END IF

      IF (LPROBTHREE)THEN

      DO 50 i=idxlo(1),idxhi(1)

      x=0.32-xx(i)
      if (xx(i).ge.0.32)x=xx(i)-0.32
      if (x.le.small)x=small

      srcbm(i)=(-2/(4*PI*(x**2)))*exp(-2.0*x)
      srcpone(i)=(-1/(4*PI))*((3*(2**2))/x)*exp(-1.732*2*x)
      qfluxbbm(i)=(1/(4*PI*(x**2)))*exp(-2.0*x)

c      write(6,*)x,srcbm(i),-src(i,j,k)
c       write(6,*)x,qfluxbbm(i),volq(i,j,k)

50    continue

      DO 60 jj=idxlo(2),idxhi(2)
         write(6,*)abs(qfluxw(ii,jj,k))
60    continue

      END IF
      
      RETURN
      end














