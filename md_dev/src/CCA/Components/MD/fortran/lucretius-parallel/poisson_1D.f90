module poisson_1D
implicit none
real(8) , allocatable :: Poisson_Matrix(:,:),poisson_field(:), poisson_A_field(:),poisson_Adip_field(:)

public :: setup_poisson_MATRIX
public :: poisson_field_eval

contains
subroutine setup_poisson_MATRIX
!Dirichet and Newman boundary conditions
use profiles_data, only : N_BINS_ZZ
use array_math, only : invmat
implicit none
integer i,j,N

N=N_BINS_ZZ
allocate(Poisson_Matrix(0:N+1,0:N+1))
allocate(poisson_field(0:N+1),poisson_A_field(0:N+1),poisson_Adip_field(0:N+1))
do i = 0, N+1
 do j = i+1, N+1
   if (iabs(i-j) > 1) then
      Poisson_Matrix(i,j) = 0.0d0
   else if (iabs(i-j) == 1) then
      Poisson_Matrix(i,j) = -1.0d0
   endif
   Poisson_Matrix(j,i) = Poisson_Matrix(i,j)
 enddo
 Poisson_Matrix(i,i) = 2.0d0
enddo
Poisson_Matrix(0,:) = 0.0d0 ; Poisson_Matrix(0,0) = 1.0d0
Poisson_Matrix(N+1,N+1) = 1.0d0
call invmat(Poisson_Matrix(0:N+1,0:N+1),N+2,N+2)

end subroutine setup_poisson_MATRIX

subroutine poisson_field_eval
use sim_cel_data
use physical_constants, only : Volt_to_internal_field,Red_Vacuum_EL_permitivity,&
                               Red_Vacuum_EL_permitivity_4_Pi,Vacuum_EL_permitivity,&
                               electron_charge,unit_length              
use profiles_data, only : N_BINS_ZZ, zp1_mol,zp1_atom
use CTRLs_data, only : l_DIP_CTRL

implicit none
integer i,j,N
real(8), allocatable :: z(:),q(:),fi(:),Aq(:),Afi(:)
real(8), allocatable :: sum_dipole(:)
real(8) h,dv,ct,s1
! need to added more when dipoles will be used
N = N_BINS_ZZ
allocate(Z(N),q(0:N+1),fi(0:N+1),Aq(0:N+1),Afi(0:N+1))   ;   
allocate(sum_dipole(0:N+1)); sum_dipole=0.0d0
do i = 1, N
  z(i) = dble(i-1)/dble(N)*sim_cel(9)  ! in Amstrom  z(i) cames in as adimentional from 0 to 1.
enddo
h=(z(3)-z(1))*0.5d0
ct=dsqrt(Red_Vacuum_EL_permitivity_4_Pi)*electron_charge/Vacuum_EL_permitivity /unit_length
if(l_DIP_CTRL) then
  do i = 2, N
   s1=0.0d0
   do j = 1, i
     s1 = s1 + zp1_atom(j)%p_dipole(3) + zp1_atom(j)%g_dipole(3)
   enddo
   sum_dipole(i) = s1
  enddo ! i=2,N
  sum_dipole(1) = sum_dipole(2)  
  sum_dipole(N+1) = sum_dipole(N)
  do i = 1, N
   Aq(i) =  zp1_atom(i)%p_charge(1) + zp1_atom(i)%g_charge(1) 
  enddo
else  ! l_DIP_CTRL
  do i = 1, N
   Aq(i) =  zp1_atom(i)%p_charge(1) + zp1_atom(i)%g_charge(1) 
  enddo
endif

  do i = 1, N
    q(i) = zp1_mol(i)%p_charge(1) + zp1_mol(i)%g_charge(1)
  enddo

  q(0) = 0.0d0     ! Dirichet boundary condition
  q(N+1) = q(N)    ! Newman   boundary condition
  Aq(0) = 0.0d0     ! Dirichet boundary condition
  Aq(N+1) = Aq(N)    ! Newman   boundary condition
  dv = Volume/dble(N)
  q = q * h**2 * ct / dV
 Aq = Aq* h**2 * ct / dV
do i = 1, N
  fi(i) = dot_product(Poisson_Matrix(i,:),q(:))
  Afi(i) = dot_product(Poisson_Matrix(i,:),Aq(:))
enddo

poisson_field = fi
poisson_A_field = Afi
poisson_Adip_field = sum_dipole * h**2 * ct / dV
deallocate(Z,q,fi,Afi,Aq)
deallocate(sum_dipole)
end subroutine poisson_field_eval

end module poisson_1D
