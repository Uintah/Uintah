! This module is token from dl-poly. 
! There are so many 3D-fft I really didn't felt like I have to rewrite them
! SO I just took one (any one, no particular reason for coosing dl-poly one)
! FFT is NOT paralelized 

! this one is mmodified such that the OZ argument is not Pi2*i*j/N but onnly i*j/N (viu instead of omega=2*Pi*miu)

module fft_3D_4_2D
 implicit none

 contains

      subroutine dlpfft3_4_2D(ind,isw,ndiv1,ndiv2,ndiv3,key1,key2,key3,ww1,ww2,ww3,aaa)


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
      integer, intent(IN) :: ndiv1,ndiv2,ndiv3
 
      logical lkx,lky,lkz
      integer key1(ndiv1),key2(ndiv2),key3(ndiv3)
      complex(8) ww1(ndiv1),ww2(ndiv2),ww3(ndiv3)
      complex(8) ttt,aaa(ndiv1,ndiv2,ndiv3)
      integer, save :: nu1,nu2,nu3
      integer idm
      integer kkk,iii,jjj,ii,kk,jj,i,j,k,jj2,jjj2,num,kk1,k12,l
      real(8) tpi,arg
       

      data tpi/6.283185307179586d0/

      if(ind.gt.0)then

!     check FFT array dimensions

        idm=1 ; lkx=.true.  ;  lky=.true.  ;  lkz=.true.

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
          if(idm.eq.ndiv3)then
            lkz=.false.
            nu3=i
          endif
        enddo ! i=1,30
        
        if(lkx.or.lky.or.lkz)then
          write(*,*)'error -dl-poly FFT array not 2**N',ndiv1,ndiv2,ndiv3
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

        do kkk=1,ndiv3
          iii=0
          jjj=kkk-1
          do j=1,nu3
            jj2=jjj/2
            iii=2*(iii-jj2)+jjj
            jjj=jj2
          enddo
          key3(kkk)=iii+1
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
        
        ww3(1)=(1.d0,0.d0)
        do i=1,ndiv3/2
          arg=(1.0d0/dble(ndiv3))*dble(i)
          ww3(i+1)=cmplx(cos(arg),sin(arg),kind=8)
          ww3(ndiv3+1-i)=conjg(ww3(i+1))
        enddo

        return

      endif ! ind ==0

!     take conjugate of exponentials if required
      
      if(isw.lt.0)then
          ww1(1:ndiv1)=conjg(ww1(1:ndiv1))
          ww2(1:ndiv2)=conjg(ww2(1:ndiv2))
          ww3(1:ndiv3)=conjg(ww3(1:ndiv3))
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
              do k=1,ndiv3
                ttt=aaa(k12,j,k)*ww1(iii)
                aaa(k12,j,k)=aaa(kk1,j,k)-ttt
                aaa(kk1,j,k)=aaa(kk1,j,k)+ttt
              enddo  !  k=1,ndiv3
            enddo    ! j=1,ndiv2
            kkk=kkk+1
          enddo  !   i=1,num
          kkk=kkk+num
        enddo  !  while(kkk.lt.ndiv1)
        kkk=0
        num=num/2
      enddo !  l=1,nu1

!     unscramble the fft using bit address array

print*, '0: ',aaa(1,1,1)

      
      do kkk=1,ndiv1
        iii=key1(kkk)
        if(iii.gt.kkk)then
          do j=1,ndiv2
            do k=1,ndiv3
              ttt=aaa(kkk,j,k)
              aaa(kkk,j,k)=aaa(iii,j,k)
              aaa(iii,j,k)=ttt
            enddo !k=1,ndiv3
          enddo   !j=1,ndiv2
        endif     !(iii.gt.kkk)
      enddo       !kkk=1,ndiv1

!     perform fourier transform in Y direction
     

print*, '1: ',aaa(1,1,1)
 
      kkk=0
      num=ndiv2/2
      do l=1,nu2
        do while(kkk.lt.ndiv2)
          do i=1,num
            iii=key2(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,ndiv1
              do k=1,ndiv3
                ttt=aaa(j,k12,k)*ww2(iii)
                aaa(j,k12,k)=aaa(j,kk1,k)-ttt
                aaa(j,kk1,k)=aaa(j,kk1,k)+ttt
              enddo
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
            do k=1,ndiv3
              ttt=aaa(j,kkk,k)
              aaa(j,kkk,k)=aaa(j,iii,k)
              aaa(j,iii,k)=ttt
            enddo
          enddo
        endif
      enddo

print*, '2: ',aaa(1,1,1)

!     perform fourier transform in Z direction
      
      kkk=0
      num=ndiv3/2
      do l=1,nu3
        do while(kkk.lt.ndiv3)
          do i=1,num
            iii=key3(kkk/num+1)
            kk1=kkk+1
            k12=kk1+num
            do j=1,ndiv1
              do k=1,ndiv2
                ttt=aaa(j,k,k12)*ww3(iii)
                aaa(j,k,k12)=aaa(j,k,kk1)-ttt
                aaa(j,k,kk1)=aaa(j,k,kk1)+ttt
              enddo
            enddo
            kkk=kkk+1
          enddo
          kkk=kkk+num
        enddo
        kkk=0
        num=num/2
      enddo

!     unscramble the fft using bit address array
      
      do kkk=1,ndiv3
        iii=key3(kkk)
        if(iii.gt.kkk)then
          do j=1,ndiv1
            do k=1,ndiv2
              ttt=aaa(j,k,kkk)
              aaa(j,k,kkk)=aaa(j,k,iii)
              aaa(j,k,iii)=ttt
            enddo
          enddo
        endif
      enddo

print*, '3: ',aaa(1,1,1)

!     restore exponentials to unconjugated values if necessary
      
      if(isw.lt.0)then
         ww1(1:ndiv1) = conjg(ww1(1:ndiv1))
         ww2(1:ndiv2) = conjg(ww2(1:ndiv2))
         ww3(1:ndiv3) = conjg(ww3(1:ndiv3))
      endif ! isw.lt.0
    
  end subroutine  dlpfft3_4_2D
end module fft_3D_4_2D

