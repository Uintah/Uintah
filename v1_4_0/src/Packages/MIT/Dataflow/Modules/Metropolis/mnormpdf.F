      subroutine mnormpdf(var,xmean,v,pdf,nvar)
      
c     Calculates pdf(var) for a multivariate t distribution, 
c     gamma((v+m)/2) / gamma(v/2)/ sqrt(det(sigma)*(v*m)^m)*(1+1/v*X'invVX)^(-(v+m)/2)
c     mean =xmean
c       variance =v
	
      integer NVAR, LDFAC,N
      PARAMETER (PI=3.141592654)
      INTEGER IPVT(nvar)
      
      double precision  V(NVAR,NVAR), VINV(NVAR,NVAR)
      double precision  X(NVAR,1),  XVINV(1,NVAR)
      double precision	VAR(NVAR,1), XMEAN(NVAR,1), XVINVX(1,1)
      double precision  DET1,DET2,FAC(nvar,nvar),detv,pdf
      double precision  g1,g2,g3,g4,g5  
      double precision a(nvar,nvar),z(nvar)
      double precision rcond
      integer job
      double precision work(nvar), det(2)
      
      character*1 transa, transb
      double precision alpha, beta
      
      integer ifault
      integer m
      
      real temp
      
      m=nvar
      
      do i=1,nvar	
         X(i,1)=VAR(i,1)-XMEAN(i,1)
      enddo
      
c     calculate vinv=inverse(V)
      do i=1,nvar
         do j=1,nvar
            vinv(i,j)=v(i,j)  
         enddo
      enddo	
      call dgeco(vinv,nvar,nvar,ipvt,rcond,z)
      call dgedi(vinv,nvar,nvar,ipvt,det,work,01)
      
c     calculate tranpose(X) * inv(V) 
      transa='T'
      transb='N'
      alpha=1.d0
      beta=0.d0
      CALL DGEMM (TRANSA, TRANSB,1, nvar, nvar, ALPHA, X, nvar, VINV, 
     1     nvar, BETA, XVINV, 1)
      
      
c     calculate [transpose(x)*inv(V)] * X
c     (1* nvar)         * (nvar*1)
      
      
      
      transa='N'
      transb='N'
      CALL DGEMM (TRANSA, TRANSB, 1, 1, nvar, ALPHA, XVINV, 1, X, 
     1     nvar, BETA, XVINVX, 1)
      
c     determinat of V
      
      do i=1,nvar
         do j=1,nvar
            a(i,j)=v(i,j)
         enddo
      enddo
      
      job=10
      call dgeco(a,nvar,nvar,ipvt,rcond,z)
      call dgedi(a,nvar,nvar,ipvt,det,work,job)
      
      detv= DET(1)*10.0**DET(2)
      
      pdf=-dble(nvar)/2.d0*dlog(2.d0*PI)-1.d0/2.d0*dlog(detv)
     1          -1.d0/2.d0*(xvinvx(1,1))
      
      return
      
      end


