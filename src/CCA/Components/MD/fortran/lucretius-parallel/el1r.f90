include  '../array-math.f90'
include '../min.f90'

  module data_data
   integer Nel  ! electrode atoms
   integer Na   ! test charges
   integer NN
   real(8), allocatable :: xyz(:,:)
   logical, allocatable :: is_distributed(:) 
   real(8), allocatable :: q(:)
   integer, allocatable :: layer(:)
   real(8), allocatable :: pot(:),force(:,:)
   real(8) box(3)

 real(8), parameter :: Vacuum_EL_permitivity_4_Pi=1.112650056d-10 !F/m
 real(8), parameter :: electron_charge=1.60217733d-19 !C
 real(8), parameter :: unit_length = 1.0d-10 !Amstrom
 integer, parameter :: MX_iter = 30
 real(8), parameter :: Pi = 3.14159265358979d0
 real(8), parameter ::  temp_cvt_en_ALL = 1.0d0/100.0d0 
 real(8) temp_cvt_en

 real(8) :: alpha = 50.0d0
 real(8) :: dummy = 1.0d0
 real(8) :: alpha_broyden = 0.3d-7
 real(8) :: eta

 real(8) ewald_alpha, ewald_beta, ewald_gamma
 integer K_MAX_X,K_MAX_Y,K_MAX_Z

 real(8) :: potential = 0.0d0
 real(8) ::  enpot = 0.0d0
  contains 
  subroutine allocate_them
    allocate(is_distributed(Nel+Na)) ; is_distributed=.false.
    allocate(q(Nel+Na))
    allocate(xyz(Nel+Na,3))
    allocate(force(Nel+Na,3))
    allocate(layer(Nel+Na))
    allocate(pot(Nel+na))
  end subroutine allocate_them
  end module data_data

 module compute_module
 use data_data
 implicit none
 contains
 subroutine real_ew
 integer i,j,k,NDIM
 real(8) En, poti,t(3),rij
 real(8), allocatable :: local_pot(:)
 allocate(local_pot(NN))

 NDIM = 2
! normal charge charge
  potential = 0.0d0
  do i = 1, Na
  poti = 0.0d0
   do j = i+1,Na
     t = xyz(i,:)-xyz(j,:)
     t(1:NDIM) = t(1:NDIM) - ANINT(t(1:NDIM)/box(1:NDIM))*box(1:NDIM) ! carefful to have enought vacuum
     rij = dsqrt(dot_product(t,t))
     En = q(i)*q(j)*erfc(ewald_alpha*rij)/rij
     pot(j) = pot(j)+En*0.5d0
     poti = poti + En*0.5d0
     potential = potential + En
   enddo 
   pot(i) = pot(i) + poti
  enddo

! distributed(electrode) charge charge
  do i =Na+1,NN
   poti = 0.0d0
   do j = i+1,NN
      t = xyz(i,:)-xyz(j,:)
      t(1:NDIM) = t(1:NDIM) - ANINT(t(1:NDIM)/box(1:NDIM))*box(1:NDIM) ! carefful to have enought vacuum
      rij = dsqrt(dot_product(t,t))
      En = q(i)*q(j)*erfc(ewald_beta*rij)/rij
      pot(j) = pot(j)+En*0.5d0
      poti = poti + En*0.5d0
      potential = potential + En
   enddo
   pot(i) = pot(i) + poti
  enddo

! mixed electrode-testcharge

  do i = Na+1,NN
   poti = 0.0d0
   do j = 1,Na
    t = xyz(i,:)-xyz(j,:)
    t(1:NDIM) = t(1:NDIM) - ANINT(t(1:NDIM)/box(1:NDIM))*box(1:NDIM) ! carefful to have enought vacuum
    rij = dsqrt(dot_product(t,t))
    En = q(i)*q(j)*erfc(ewald_gamma*rij)/rij
    pot(j) = pot(j)+En*0.5d0
    poti = poti + En*0.5d0
    potential = potential + En
   enddo
   pot(i) = pot(i) + poti
  enddo

 potential = potential/Vacuum_EL_permitivity_4_Pi
 pot = pot / Vacuum_EL_permitivity_4_Pi
 print*, ' real potential = ', potential*temp_cvt_en, sum(pot)*temp_cvt_en
 enpot = enpot + potential 
 deallocate(local_pot)
 end subroutine real_ew

 subroutine Fourier_ew_0
 integer i,j,k
 real(8) CC1,CC2,CC3, En,zij,poti,Area,iArea
 real(8), allocatable :: local_pot(:)
 allocate(local_pot(NN))
 
   CC1 = dsqrt(Pi)/Ewald_alpha
   CC2 = dsqrt(Pi)/Ewald_beta
   CC3 = dsqrt(Pi)/Ewald_gamma
   Area = box(1)*box(2) ; iArea = 1.0d0/Area
   potential = 0.0d0
   local_pot = 0.0d0
   do i = 1, Na
   poti = 0.0d0
   do j = i+1,Na
     zij = xyz(i,3)-xyz(j,3)
     En = 2.0d0*q(i)*q(j)*(CC1*dexp(-(zij*Ewald_alpha)**2)+zij*Pi*erf(zij*Ewald_alpha))*iArea
     local_pot(j) = local_pot(j)+En*0.5d0
     poti = poti + En*0.5d0
     potential = potential + En
   enddo
   local_pot(i) = local_pot(i) + poti
  enddo
print*, '1 potential local_pot=',potential, sum(local_pot)
! distributed(electrode) charge charge
  do i =Na+1,NN
  poti = 0.0d0
   do j = i+1,NN
      zij = xyz(i,3)-xyz(j,3)
      En = 2.0d0*q(i)*q(j)*(CC2*dexp(-(zij*Ewald_beta)**2)+zij*Pi*erf(zij*Ewald_beta))*iArea
      local_pot(j) = local_pot(j)+En*0.5d0
      poti = poti + En*0.5d0
      potential = potential + En
   enddo
   local_pot(i) = local_pot(i) + poti
  enddo
print*, '2 potential local_pot=',potential, sum(local_pot)

! mixed electrode-testcharge

  do i = Na+1,NN
  poti = 0.0d0
   do j = 1,Na
    zij = xyz(i,3)-xyz(j,3)
    En = 2.0d0*q(i)*q(j)*(CC2*dexp(-(zij*Ewald_gamma)**2)+zij*Pi*erf(zij*Ewald_gamma))*iArea
    local_pot(j) = local_pot(j)+En*0.5d0
    poti = poti + En*0.5d0
    potential = potential + En
   enddo
   local_pot(i) = local_pot(i) + poti
  enddo
print*, '3 potential local_pot=',potential, sum(local_pot)

 potential = potential/Vacuum_EL_permitivity_4_Pi
 local_pot = local_pot / Vacuum_EL_permitivity_4_Pi
 pot = pot + local_pot
 print*, ' real zero Fourier = ', potential*temp_cvt_en, sum(local_pot)*temp_cvt_en

 enpot = enpot + potential 
 deallocate(local_pot)
 end subroutine Fourier_ew_0

 end module compute_module


  program main
  use data_data
  use compute_module
  implicit none
  integer i,j,k
  real(8) bla

    open(unit=10,file='config')
    read(10,*) Nel,Na
    NN = Nel+Na     ; temp_cvt_en = temp_cvt_en_ALL/dble(NN)
    call allocate_them
    read(10,*) box
    do i = 1, Na
     read(10,*) xyz(Nel+i,:),q(Nel+i)
     is_distributed(Nel+i)=.false.
    enddo 
    do i = 1, Nel
     read(10,*) xyz(i,:),layer(i),q(i)
     is_distributed(i)=.true.
    enddo
    close(10)

    eta = 0.5d0
    ewald_alpha = 0.25d0
    ewald_beta = eta*ewald_alpha/dsqrt(eta**2+2.0d0*ewald_alpha**2)
    ewald_gamma = eta*ewald_alpha/dsqrt(eta**2+ewald_alpha**2)

    call real_ew
    call Fourier_ew_0 
  end program main 
