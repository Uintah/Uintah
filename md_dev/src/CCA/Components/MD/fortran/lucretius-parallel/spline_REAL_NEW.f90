module beta_spline_new
! evaluated more eficiently
 public :: beta_spline_pp
 public :: beta_spline_pp_dd
 public :: beta_spline_pp_dd_2

contains

 subroutine beta_spline_pp
 use sizes_data, only : Natoms
 use Ewald_data, only : order_spline_xx,order_spline_yy,order_spline_zz
 use variables_smpe, only : nfftx,nffty,nfftz,tx,ty,tz, &
                    spline2_REAL_pp_x, spline2_REAL_pp_y,spline2_REAL_pp_z
 use all_atoms_data, only : xx,yy,zz,xxx,yyy,zzz
 use sim_cel_data, only : i_boundary_CTRL

 implicit none
 real(8),allocatable :: buffer3(:,:),buffer1(:)
 real(8) aaa,bbb,ccc,iki
 integer i,j,k

 allocate(buffer3(Natoms,3))
 allocate(buffer1(Natoms))

  if (i_boundary_CTRL == 1) then
   do i = 1, Natoms
    buffer3(i,1) = tx(i)-int(tx(i)) ! tt are the reduced coordinates
    buffer3(i,2) = ty(i)-int(ty(i))
    buffer3(i,3) = zzz(i)-int(zzz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
   enddo
  else 
    do i = 1, Natoms
     buffer3(i,1) = tx(i)-int(tx(i)) ! tt are the reduced coordinates
     buffer3(i,2) = ty(i)-int(ty(i))
     buffer3(i,3) = tz(i)-int(tz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
    enddo
   endif

 do i = 1, Natoms
   spline2_REAL_pp_x(i,1) = buffer3(i,1)
   spline2_REAL_pp_y(i,1) = buffer3(i,2)
   spline2_REAL_pp_z(i,1) = buffer3(i,3)
   spline2_REAL_pp_x(i,2) = 1.0d0 - spline2_REAL_pp_x(i,1)
   spline2_REAL_pp_y(i,2) = 1.0d0 - spline2_REAL_pp_y(i,1)
   spline2_REAL_pp_z(i,2) = 1.0d0 - spline2_REAL_pp_z(i,1)
 enddo

 do k = 3, order_spline_xx
        iki  = 1.0d0/dble(k-1)
        spline2_REAL_pp_x(1:Natoms,k) = 0.0d0
        do j=k,2,-1
           do i = 1, Natoms
            aaa=buffer3(i,1)+dble(j-1)
  spline2_REAL_pp_x(i,j)=(aaa*spline2_REAL_pp_x(i,j)+(dble(k)-aaa)*spline2_REAL_pp_x(i,j-1))*iki
           enddo
        enddo ! j=k,2,-1
        do i = 1, Natoms
         spline2_REAL_pp_x(i,1)=buffer3(i,1) * spline2_REAL_pp_x(i,1)*iki
        enddo
      enddo ! k=3,N_splines

      do k = 3, order_spline_yy
        iki  = 1.0d0/dble(k-1)
        spline2_REAL_pp_y(1:Natoms,k) = 0.0d0
        do j=k,2,-1
           do i = 1, Natoms
            bbb=buffer3(i,2)+dble(j-1)
  spline2_REAL_pp_y(i,j)=(bbb*spline2_REAL_pp_y(i,j)+(dble(k)-bbb)*spline2_REAL_pp_y(i,j-1))*iki
           enddo
        enddo ! j=k,2,-1
        do i = 1, Natoms
         spline2_REAL_pp_y(i,1)=buffer3(i,2) * spline2_REAL_pp_y(i,1)*iki
        enddo
      enddo ! k=3,N_splines

        do k = 3, order_spline_zz
        iki  = 1.0d0/dble(k-1)
        spline2_REAL_pp_z(1:Natoms,k) = 0.0d0
        do j=k,2,-1
           do i = 1, Natoms
            ccc=buffer3(i,3)+dble(j-1)
  spline2_REAL_pp_z(i,j)=(ccc*spline2_REAL_pp_z(i,j)+(dble(k)-ccc)*spline2_REAL_pp_z(i,j-1))*iki
           enddo
        enddo ! j=k,2,-1
        do i = 1, Natoms
         spline2_REAL_pp_z(i,1)=buffer3(i,3) * spline2_REAL_pp_z(i,1)*iki
        enddo
      enddo ! k=3,N_splines


     deallocate(buffer3)
! I do the i<->N-i flipping to have the same arrangement as in SMPE
     
     do i = 1, order_spline_xx/2
       buffer1(:) = spline2_REAL_pp_x(:,i)
       spline2_REAL_pp_x(:,i) = spline2_REAL_pp_x(:,order_spline_xx-i+1)
       spline2_REAL_pp_x(:,order_spline_xx-i+1) = buffer1(:)
     enddo
     do i = 1, order_spline_yy/2
       buffer1(:) = spline2_REAL_pp_y(:,i)
       spline2_REAL_pp_y(:,i) = spline2_REAL_pp_y(:,order_spline_yy-i+1)
       spline2_REAL_pp_y(:,order_spline_yy-i+1) = buffer1(:)
     enddo
     do i = 1, order_spline_zz/2
       buffer1(:) = spline2_REAL_pp_z(:,i)
       spline2_REAL_pp_z(:,i) = spline2_REAL_pp_z(:,order_spline_zz-i+1)
       spline2_REAL_pp_z(:,order_spline_zz-i+1) = buffer1(:)
     enddo
     deallocate(buffer1)
 
 end subroutine beta_spline_pp


 subroutine beta_spline_pp_dd
 use sizes_data, only : Natoms
 use Ewald_data, only : order_spline_xx,order_spline_yy,order_spline_zz
 use variables_smpe, only : nfftx,nffty,nfftz,tx,ty,tz, &
                    spline2_REAL_dd_x, spline2_REAL_dd_y,spline2_REAL_dd_z,&
                    spline2_REAL_pp_x, spline2_REAL_pp_y,spline2_REAL_pp_z
 use all_atoms_data, only : xx,yy,zz,xxx,yyy,zzz
 use sim_cel_data, only : i_boundary_CTRL

 implicit none
 real(8),allocatable :: buffer3(:,:),buffer1(:)
 real(8) aaa,bbb,ccc,iki
 integer i,j,k

 allocate(buffer3(Natoms,3))

  if (i_boundary_CTRL == 1) then
   do i = 1, Natoms
    buffer3(i,1) = tx(i)-int(tx(i)) ! tt are the reduced coordinates
    buffer3(i,2) = ty(i)-int(ty(i))
    buffer3(i,3) = zzz(i)-int(zzz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
   enddo
  else
    do i = 1, Natoms
     buffer3(i,1) = tx(i)-int(tx(i)) ! tt are the reduced coordinates
     buffer3(i,2) = ty(i)-int(ty(i))
     buffer3(i,3) = tz(i)-int(tz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
    enddo
   endif

   spline2_REAL_dd_x(:,1)= 1.0d0; spline2_REAL_dd_y(:,1)= 1.0d0;spline2_REAL_dd_z(:,1)= 1.0d0 
   spline2_REAL_dd_x(:,2)=-1.0d0; spline2_REAL_dd_y(:,2)=-1.0d0;spline2_REAL_dd_z(:,2)=-1.0d0
   spline2_REAL_pp_x(:,1) = buffer3(:,1) ; 
   spline2_REAL_pp_y(:,1) = buffer3(:,2) ;
   spline2_REAL_pp_z(:,1) = buffer3(:,3) ;
   do i = 1, Natoms
       spline2_REAL_pp_x(i,2) = 1.0d0 - spline2_REAL_pp_x(i,1)
       spline2_REAL_pp_y(i,2) = 1.0d0 - spline2_REAL_pp_y(i,1)
       spline2_REAL_pp_z(i,2) = 1.0d0 - spline2_REAL_pp_z(i,1)
   enddo

   do k = 3,order_spline_xx
   iki = 1.0d0/dble(k-1)
        spline2_REAL_pp_x(1:Natoms,k) = 0.0d0
        do j=k,2,-1
           if(k.eq.order_spline_xx) then
           do i = 1, Natoms
              spline2_REAL_dd_x(i,j)=spline2_REAL_pp_x(i,j)-spline2_REAL_pp_x(i,j-1)
           enddo
           endif
           do i = 1, Natoms
            aaa=buffer3(i,1) +  dble(j-1)
  spline2_REAL_pp_x(i,j)=(aaa*spline2_REAL_pp_x(i,j)+(dble(k)-aaa)*spline2_REAL_pp_x(i,j-1)) * iki
           enddo
        enddo ! j=k,2,-1

        if (k.eq.order_spline_xx) then
         spline2_REAL_dd_x(1:Natoms,1)= spline2_REAL_pp_x(1:Natoms,1)
        endif

        do i = 1, Natoms
         spline2_REAL_pp_x(i,1)=buffer3(i,1)* spline2_REAL_pp_x(i,1) * iki
        enddo

      enddo ! k=3,N_splines

!

   do k = 3,order_spline_yy
   iki = 1.0d0/dble(k-1)
        spline2_REAL_pp_y(1:Natoms,k) = 0.0d0
        do j=k,2,-1
           if(k.eq.order_spline_yy) then
           do i = 1, Natoms
              spline2_REAL_dd_y(i,j)=spline2_REAL_pp_y(i,j)-spline2_REAL_pp_y(i,j-1)
           enddo
           endif
           do i = 1, Natoms
            aaa=buffer3(i,2) +  dble(j-1)
  spline2_REAL_pp_y(i,j)=(aaa*spline2_REAL_pp_y(i,j)+(dble(k)-aaa)*spline2_REAL_pp_y(i,j-1)) * iki
           enddo
        enddo ! j=k,2,-1

        if (k.eq.order_spline_yy) then
         spline2_REAL_dd_y(1:Natoms,1)= spline2_REAL_pp_y(1:Natoms,1)
        endif

        do i = 1, Natoms
         spline2_REAL_pp_y(i,1)=buffer3(i,2)* spline2_REAL_pp_y(i,1) * iki
        enddo

      enddo ! k=3,N_splines

!

   do k = 3,order_spline_zz
   iki = 1.0d0/dble(k-1)
        spline2_REAL_pp_z(1:Natoms,k) = 0.0d0
        do j=k,2,-1
           if(k.eq.order_spline_zz) then
           do i = 1, Natoms
              spline2_REAL_dd_z(i,j)=spline2_REAL_pp_z(i,j)-spline2_REAL_pp_z(i,j-1)
           enddo
           endif
           do i = 1, Natoms
            aaa=buffer3(i,3) +  dble(j-1)
  spline2_REAL_pp_z(i,j)=(aaa*spline2_REAL_pp_z(i,j)+(dble(k)-aaa)*spline2_REAL_pp_z(i,j-1)) * iki
           enddo
        enddo ! j=k,2,-1

        if (k.eq.order_spline_zz) then
         spline2_REAL_dd_z(1:Natoms,1)= spline2_REAL_pp_z(1:Natoms,1)
        endif

        do i = 1, Natoms
         spline2_REAL_pp_z(i,1)=buffer3(i,3)* spline2_REAL_pp_z(i,1) * iki
        enddo

      enddo ! k=3,N_splines

 
  deallocate(buffer3)
  allocate(buffer1(Natoms))
     do i = 1, order_spline_xx/2
       buffer1(:) = spline2_REAL_pp_x(:,i)
       spline2_REAL_pp_x(:,i) = spline2_REAL_pp_x(:,order_spline_xx-i+1)
       spline2_REAL_pp_x(:,order_spline_xx-i+1) = buffer1(:)
       buffer1(:) = spline2_REAL_dd_x(:,i)
       spline2_REAL_dd_x(:,i) = spline2_REAL_dd_x(:,order_spline_xx-i+1)
       spline2_REAL_dd_x(:,order_spline_xx-i+1) = buffer1(:)
     enddo
     do i = 1, order_spline_yy/2
       buffer1(:) = spline2_REAL_pp_y(:,i)
       spline2_REAL_pp_y(:,i) = spline2_REAL_pp_y(:,order_spline_yy-i+1)
       spline2_REAL_pp_y(:,order_spline_yy-i+1) = buffer1(:)
       buffer1(:) = spline2_REAL_dd_y(:,i)
       spline2_REAL_dd_y(:,i) = spline2_REAL_dd_y(:,order_spline_yy-i+1)
       spline2_REAL_dd_y(:,order_spline_yy-i+1) = buffer1(:)
     enddo
     do i = 1, order_spline_zz/2
       buffer1(:) = spline2_REAL_pp_z(:,i)
       spline2_REAL_pp_z(:,i) = spline2_REAL_pp_z(:,order_spline_zz-i+1)
       spline2_REAL_pp_z(:,order_spline_zz-i+1) = buffer1(:)
       buffer1(:) = spline2_REAL_dd_z(:,i)
       spline2_REAL_dd_z(:,i) = spline2_REAL_dd_z(:,order_spline_zz-i+1)
       spline2_REAL_dd_z(:,order_spline_zz-i+1) = buffer1(:)
     enddo
     deallocate(buffer1)

  end subroutine beta_spline_pp_dd

  subroutine beta_spline_pp_dd_2
! second derivative is included
 use sizes_data, only : Natoms
 use Ewald_data, only : order_spline_xx,order_spline_yy,order_spline_zz
 use variables_smpe, only : nfftx,nffty,nfftz,tx,ty,tz, &
                    spline2_REAL_dd_x, spline2_REAL_dd_y,spline2_REAL_dd_z,&
                    spline2_REAL_pp_x, spline2_REAL_pp_y,spline2_REAL_pp_z,&
                    spline2_REAL_dd_2_x, spline2_REAL_dd_2_y,spline2_REAL_dd_2_z
 use all_atoms_data, only : xx,yy,zz,xxx,yyy,zzz
 use sim_cel_data, only : i_boundary_CTRL

 implicit none
 real(8),allocatable :: buffer3(:,:), epsiloni(:),buffer1(:)
 real(8) aaa,bbb,ccc,iki
 integer i,j,k

 allocate(buffer3(Natoms,3),epsiloni(Natoms))

  if (i_boundary_CTRL == 1) then
   do i = 1, Natoms
    buffer3(i,1) = tx(i)-int(tx(i)) ! tt are the reduced coordinates
    buffer3(i,2) = ty(i)-int(ty(i))
    buffer3(i,3) = zzz(i)-int(zzz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
   enddo
  else
    do i = 1, Natoms
     buffer3(i,1) = tx(i)-int(tx(i)) ! tt are the reduced coordinates
     buffer3(i,2) = ty(i)-int(ty(i))
     buffer3(i,3) = tz(i)-int(tz(i))   !beta_spline_REAL_coef(order,x,nfft,spline,spline_DERIV)
    enddo
   endif


   spline2_REAL_dd_x(:,1)= 1.0d0; spline2_REAL_dd_y(:,1)= 1.0d0;spline2_REAL_dd_z(:,1)= 1.0d0
   spline2_REAL_dd_x(:,2)=-1.0d0; spline2_REAL_dd_y(:,2)=-1.0d0;spline2_REAL_dd_z(:,2)=-1.0d0
   spline2_REAL_pp_x(:,1) = buffer3(:,1) ;
   spline2_REAL_pp_y(:,1) = buffer3(:,2) ;
   spline2_REAL_pp_z(:,1) = buffer3(:,3) ;
   do i = 1, Natoms
       spline2_REAL_pp_x(i,2) = 1.0d0 - spline2_REAL_pp_x(i,1)
       spline2_REAL_pp_y(i,2) = 1.0d0 - spline2_REAL_pp_y(i,1)
       spline2_REAL_pp_z(i,2) = 1.0d0 - spline2_REAL_pp_z(i,1)
   enddo
   
   epsiloni(:) =  spline2_REAL_pp_x(:,2) 


!

   do k = 3,order_spline_xx
   iki = 1.0d0/dble(k-1)
        spline2_REAL_pp_x(1:Natoms,k) = 0.0d0
        do j=k,2,-1
           if(k.eq.order_spline_xx) then
           do i = 1, Natoms
              spline2_REAL_dd_x(i,j)=spline2_REAL_pp_x(i,j)-spline2_REAL_pp_x(i,j-1)
           enddo
           endif
! \second order spline
   if(k.eq.order_spline_xx-1) then
   if (j > 2) then
   do i = 1, Natoms
     spline2_REAL_dd_2_x(i,j)=spline2_REAL_pp_x(i,j)-2.0d0*spline2_REAL_pp_x(i,j-1)+spline2_REAL_pp_x(i,j-2)
   enddo
   elseif (j==2) then
   do i = 1, Natoms
     spline2_REAL_dd_2_x(i,j)=spline2_REAL_pp_x(i,j)-2.0d0*spline2_REAL_pp_x(i,j-1)
   enddo
   endif
   endif
! \second order spline
           do i = 1, Natoms
            aaa=buffer3(i,1) +  dble(j-1)
  spline2_REAL_pp_x(i,j)=(aaa*spline2_REAL_pp_x(i,j)+(dble(k)-aaa)*spline2_REAL_pp_x(i,j-1)) * iki
           enddo
        enddo ! j=k,2,-1
! \second order spline
 if (k == order_spline_xx-1) then
   spline2_REAL_dd_2_x(1:Natoms,1)= spline2_REAL_pp_x(1:Natoms,1)
   spline2_REAL_dd_2_x(1:Natoms,order_spline_xx)= epsiloni(1:Natoms)
 endif
! \end second order spline
        if (k.eq.order_spline_xx) then
         spline2_REAL_dd_x(1:Natoms,1)= spline2_REAL_pp_x(1:Natoms,1)
        endif

        do i = 1, Natoms
         spline2_REAL_pp_x(i,1)=buffer3(i,1)* spline2_REAL_pp_x(i,1) * iki
! \second order spline
         epsiloni(i) = (-1.0d0+buffer3(i,1)) *  epsiloni(i) * iki
! \end second order spline
        enddo

      enddo ! k=3,N_splines

! 
!   y:
!

   epsiloni(:) =  spline2_REAL_pp_y(:,2)

   do k = 3,order_spline_yy
   iki = 1.0d0/dble(k-1)
        spline2_REAL_pp_y(1:Natoms,k) = 0.0d0
        do j=k,2,-1
           if(k.eq.order_spline_yy) then
           do i = 1, Natoms
              spline2_REAL_dd_y(i,j)=spline2_REAL_pp_y(i,j)-spline2_REAL_pp_y(i,j-1)
           enddo
           endif
! \second order spline
   if(k.eq.order_spline_yy-1) then
   if (j > 2) then
   do i = 1, Natoms
     spline2_REAL_dd_2_y(i,j)=spline2_REAL_pp_y(i,j)-2.0d0*spline2_REAL_pp_y(i,j-1)+spline2_REAL_pp_y(i,j-2)
   enddo
   elseif (j==2) then
   do i = 1, Natoms
     spline2_REAL_dd_2_y(i,j)=spline2_REAL_pp_y(i,j)-2.0d0*spline2_REAL_pp_y(i,j-1)
   enddo
   endif
   endif
! \second order spline

           do i = 1, Natoms
            aaa=buffer3(i,2) +  dble(j-1)
  spline2_REAL_pp_y(i,j)=(aaa*spline2_REAL_pp_y(i,j)+(dble(k)-aaa)*spline2_REAL_pp_y(i,j-1)) * iki
           enddo
        enddo ! j=k,2,-1
! \second order spline
 if (k == order_spline_yy-1) then
   spline2_REAL_dd_2_y(1:Natoms,1)= spline2_REAL_pp_y(1:Natoms,1)
   spline2_REAL_dd_2_y(1:Natoms,order_spline_yy)= epsiloni(1:Natoms)
 endif
! \end second order spline

        if (k.eq.order_spline_yy) then
         spline2_REAL_dd_y(1:Natoms,1)= spline2_REAL_pp_y(1:Natoms,1)
        endif

        do i = 1, Natoms
         spline2_REAL_pp_y(i,1)=buffer3(i,2)* spline2_REAL_pp_y(i,1) * iki
! \second order spline
         epsiloni(i) = (-1.0d0+buffer3(i,2)) *  epsiloni(i) * iki
! \end second order spline
        enddo

      enddo ! k=3,N_splines

   
!
!   z:
!
   epsiloni(:) =  spline2_REAL_pp_z(:,2)


   do k = 3,order_spline_zz
   iki = 1.0d0/dble(k-1)
        spline2_REAL_pp_z(1:Natoms,k) = 0.0d0
        do j=k,2,-1
           if(k.eq.order_spline_zz) then
           do i = 1, Natoms
              spline2_REAL_dd_z(i,j)=spline2_REAL_pp_z(i,j)-spline2_REAL_pp_z(i,j-1)
           enddo
           endif
! \second order spline
   if(k.eq.order_spline_zz-1) then
   if (j > 2) then
   do i = 1, Natoms
     spline2_REAL_dd_2_z(i,j)=spline2_REAL_pp_z(i,j)-2.0d0*spline2_REAL_pp_z(i,j-1)+spline2_REAL_pp_z(i,j-2)
   enddo
   elseif (j==2) then
   do i = 1, Natoms
     spline2_REAL_dd_2_z(i,j)=spline2_REAL_pp_z(i,j)-2.0d0*spline2_REAL_pp_z(i,j-1)
   enddo
   endif
   endif
! \second order spline
           do i = 1, Natoms
            aaa=buffer3(i,3) +  dble(j-1)
  spline2_REAL_pp_z(i,j)=(aaa*spline2_REAL_pp_z(i,j)+(dble(k)-aaa)*spline2_REAL_pp_z(i,j-1)) * iki
           enddo
        enddo ! j=k,2,-1

! \second order spline
 if (k == order_spline_zz-1) then
   spline2_REAL_dd_2_z(1:Natoms,1)= spline2_REAL_pp_z(1:Natoms,1)
   spline2_REAL_dd_2_z(1:Natoms,order_spline_zz)= epsiloni(1:Natoms)
 endif
! \end second order spline

        if (k.eq.order_spline_zz) then
         spline2_REAL_dd_z(1:Natoms,1)= spline2_REAL_pp_z(1:Natoms,1)
        endif

        do i = 1, Natoms
         spline2_REAL_pp_z(i,1)=buffer3(i,3)* spline2_REAL_pp_z(i,1) * iki
! \second order spline
         epsiloni(i) = (-1.0d0+buffer3(i,3)) *  epsiloni(i) * iki
! \end second order spline
        enddo

      enddo ! k=3,N_splines

  deallocate(buffer3,epsiloni)  
  allocate(buffer1(Natoms))
     do i = 1, order_spline_xx/2
       buffer1(:) = spline2_REAL_pp_x(:,i)
       spline2_REAL_pp_x(:,i) = spline2_REAL_pp_x(:,order_spline_xx-i+1)
       spline2_REAL_pp_x(:,order_spline_xx-i+1) = buffer1(:)
       buffer1(:) = spline2_REAL_dd_x(:,i)
       spline2_REAL_dd_x(:,i) = spline2_REAL_dd_x(:,order_spline_xx-i+1)
       spline2_REAL_dd_x(:,order_spline_xx-i+1) = buffer1(:)
       buffer1(:) = spline2_REAL_dd_2_x(:,i)
       spline2_REAL_dd_2_x(:,i) = spline2_REAL_dd_2_x(:,order_spline_xx-i+1)
       spline2_REAL_dd_2_x(:,order_spline_xx-i+1) = buffer1(:)
     enddo
     do i = 1, order_spline_yy/2
       buffer1(:) = spline2_REAL_pp_y(:,i)
       spline2_REAL_pp_y(:,i) = spline2_REAL_pp_y(:,order_spline_yy-i+1)
       spline2_REAL_pp_y(:,order_spline_yy-i+1) = buffer1(:)
       buffer1(:) = spline2_REAL_dd_y(:,i)
       spline2_REAL_dd_y(:,i) = spline2_REAL_dd_y(:,order_spline_yy-i+1)
       spline2_REAL_dd_y(:,order_spline_yy-i+1) = buffer1(:)
       buffer1(:) = spline2_REAL_dd_2_y(:,i)
       spline2_REAL_dd_2_y(:,i) = spline2_REAL_dd_2_y(:,order_spline_yy-i+1)
       spline2_REAL_dd_2_y(:,order_spline_yy-i+1) = buffer1(:)
     enddo
     do i = 1, order_spline_zz/2
       buffer1(:) = spline2_REAL_pp_z(:,i)
       spline2_REAL_pp_z(:,i) = spline2_REAL_pp_z(:,order_spline_zz-i+1)
       spline2_REAL_pp_z(:,order_spline_zz-i+1) = buffer1(:)
       buffer1(:) = spline2_REAL_dd_z(:,i)
       spline2_REAL_dd_z(:,i) = spline2_REAL_dd_z(:,order_spline_zz-i+1)
       spline2_REAL_dd_z(:,order_spline_zz-i+1) = buffer1(:)
       buffer1(:) = spline2_REAL_dd_2_z(:,i)
       spline2_REAL_dd_2_z(:,i) = spline2_REAL_dd_2_z(:,order_spline_zz-i+1)
       spline2_REAL_dd_2_z(:,order_spline_zz-i+1) = buffer1(:)
     enddo
     deallocate(buffer1)

  end subroutine beta_spline_pp_dd_2



end module beta_spline_new
