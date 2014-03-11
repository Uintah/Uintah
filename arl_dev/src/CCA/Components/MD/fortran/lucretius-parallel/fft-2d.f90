! modified dl-poly subroutine to do 2D 

module fft_2D
 implicit none

 contains

      subroutine dlpfft2(ind,isw,ndiv1,ndiv2,key1,key2,ww1,ww2,aaa)


!***********************************************************************
!     
!     dl-poly 3D fast fourier transform routine (in place)
!     
!     copyright daresbury laboratory 1998
!     
!     author w smith july 1998
!     
!     wl
!     2002/05/31 13:58:07
!     1.2
!     Exp
!     
!***********************************************************************
      
      integer, intent(IN) :: ind,isw
      integer, intent(IN) :: ndiv1,ndiv2
 
      logical lkx,lky,lkz
      integer key1(ndiv1),key2(ndiv2)
      complex(8) ww1(ndiv1),ww2(ndiv2)
      complex(8) ttt,aaa(ndiv1,ndiv2)
      integer, save :: nu1,nu2
      integer idm
      integer kkk,iii,jjj,ii,kk,jj,i,j,k,jj2,jjj2,num,kk1,k12,l
      real(8) tpi,arg
       

      data tpi/6.283185307179586d0/

      if(ind.gt.0)then

!     check FFT array dimensions

        idm=1 ; lkx=.true.  ;  lky=.true.  ;  

        do i=1,30
          idm=2*idm
          if(idm.eq.ndiv1)then
            lkx=.false.
            nu1=i
          endif
          if(idm.eq.ndiv2)then
            lky=.false.
            nu2=i
          endif
        enddo ! i=1,30
        
        if(lkx.or.lky)then
          write(*,*)'error -dl-poly 2D FFT array not 2**N',ndiv1,ndiv2
          stop
        endif
        
        do kkk=1,ndiv1
          iii=0
          jjj=kkk-1
          do j=1,nu1
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key1(kkk)=iii+1
        enddo

        do kkk=1,ndiv2
          iii=0
          jjj=kkk-1
          do j=1,nu2
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key2(kkk)=iii+1
        enddo

!     initialise complex exponential factors
        
        ww1(1)=(1.d0,0.d0)
        do i=1,ndiv1/2
          arg=(tpi/dble(ndiv1))*dble(i)
          ww1(i+1)=cmplx(cos(arg),sin(arg),kind=8)
          ww1(ndiv1+1-i)=conjg(ww1(i+1))
        enddo
        
        ww2(1)=(1.d0,0.d0)
        do i=1,ndiv2/2
          arg=(tpi/dble(ndiv2))*dble(i)
          ww2(i+1)=cmplx(cos(arg),sin(arg),kind=8)
          ww2(ndiv2+1-i)=conjg(ww2(i+1))
        enddo
        
        return
      endif ! ind ==0

!     take conjugate of exponentials if required
      
      if(isw.lt.0)then
          ww1(1:ndiv1)=conjg(ww1(1:ndiv1))
          ww2(1:ndiv2)=conjg(ww2(1:ndiv2))
      endif ! isw.lt.0

!     perform fourier transform in X direction
      
      kkk=0
      num=ndiv1/2
      do l=1,nu1
        do while(kkk.lt.ndiv1)
          do i=1,num
            iii=key1(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,ndiv2
                ttt=aaa(k12,j)*ww1(iii)
                aaa(k12,j)=aaa(kk1,j)-ttt
                aaa(kk1,j)=aaa(kk1,j)+ttt
            enddo    ! j=1,ndiv2
            kkk=kkk+1
          enddo  !   i=1,num
          kkk=kkk+num
        enddo  !  while(kkk.lt.ndiv1)
        kkk=0
        num=num/2
      enddo !  l=1,nu1

!     unscramble the fft using bit address array
      
      do kkk=1,ndiv1
        iii=key1(kkk)
        if(iii.gt.kkk)then
          do j=1,ndiv2
              ttt=aaa(kkk,j)
              aaa(kkk,j)=aaa(iii,j)
              aaa(iii,j)=ttt
          enddo   !j=1,ndiv2
        endif     !(iii.gt.kkk)
      enddo       !kkk=1,ndiv1

!     perform fourier transform in Y direction
      
      kkk=0
      num=ndiv2/2
      do l=1,nu2
        do while(kkk.lt.ndiv2)
          do i=1,num
            iii=key2(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,ndiv1
                ttt=aaa(j,k12)*ww2(iii)
                aaa(j,k12)=aaa(j,kk1)-ttt
                aaa(j,kk1)=aaa(j,kk1)+ttt
            enddo
            kkk=kkk+1
          enddo
          kkk=kkk+num
        enddo
        kkk=0
        num=num/2
      enddo

!     unscramble the fft using bit address array
      
      do kkk=1,ndiv2
        iii=key2(kkk)
        if(iii.gt.kkk)then
          do j=1,ndiv1
              ttt=aaa(j,kkk)
              aaa(j,kkk)=aaa(j,iii)
              aaa(j,iii)=ttt
          enddo
        endif
      enddo

!     restore exponentials to unconjugated values if necessary
      
      if(isw.lt.0)then
         ww1(1:ndiv1) = conjg(ww1(1:ndiv1))
         ww2(1:ndiv2) = conjg(ww2(1:ndiv2))
      endif ! isw.lt.0
    
  end subroutine  dlpfft2
end module fft_2D

