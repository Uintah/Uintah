module ewald_2D_k0_Q_DIP_module
 implicit none
 real(8), private, allocatable , save :: qq(:), bv(:,:),z_grid(:),db(:,:)
 integer, private :: order, Ngrid, N_size_qq
 private :: set_energy_2D_k0_Q_DIP
 public :: driver_ewald_2D_k0_Q_DIP 
 contains

 subroutine driver_ewald_2D_k0_Q_DIP
! 1G means that all gaussian charges have the same width parameter 
! that is an extrodinary simplification of the problem
   use Ewald_data ! order_spline_zz_k0, n_grid_zz_k0
   use spline_z_k0_module
   use energies_data ! for debug
   implicit none
   call initialize
   call set_energy_2D_k0_Q_DIP
   call finalize
   contains 
   subroutine initialize
      order = order_spline_zz_k0
      Ngrid = n_grid_zz_k0
      N_size_qq = Ngrid+order+1
      allocate(qq(1:N_size_qq), bv(1:Ngrid+1,1:2),z_grid(1:Ngrid))
      allocate(db(1:Ngrid+1,1:2))
      call get_z_grid(order,Ngrid,z_grid)
      call get_qq_coordinate(order,Ngrid,z_grid,qq)
   end subroutine initialize
   subroutine finalize
      deallocate(qq,bv,z_grid)
      deallocate(db)
   end subroutine finalize
 end subroutine driver_ewald_2D_k0_Q_DIP
 
! include 'Ew_2D_k0_Q_SLOW.f90'

 subroutine set_energy_2D_k0_Q_DIP
    use spline_z_k0_module
    use array_math
    use sim_cel_data
    use ALL_atoms_data, only : xx,yy,zz,all_charges,Natoms, fzz, &
                               is_charge_distributed, i_type_atom, zzz,all_dipoles_zz
    use Ewald_data
    use connectivity_ALL_data, only : list_excluded_HALF,size_list_excluded_HALF,MX_excluded
    use boundaries, only : periodic_images_ZZ
    use energies_data
    use profiles_data, only : atom_profile, l_need_2nd_profile
    use sizes_data, only : N_TYPE_ATOMS
    use stresses_data

    implicit none
    real(8), parameter :: sqrt_Pi = 1.77245385090552d0
    real(8), parameter :: Pi = 3.14159265358979d0
    real(8), parameter :: Pi2 = 2.0d0*Pi
    integer i,j,k,n,kk2,om2,kkk,i1
    integer jlim1,jlim2
    real(8) i_Area,CC1,zk,x,qi,xxx, ff,ff1,erf_field, sum_erf_field,field_1,zij,qij,qj
    real(8) sum_stress_field, local_stress_zz, szz,szz_1
    real(8), allocatable :: MAT(:,:), alp(:),alp_1(:),field_grid(:),field_grid_1(:)
    real(8), allocatable :: field_force(:), field_force_1(:)
    real(8), allocatable :: stress_field(:), stress_field_1(:)
    real(8), allocatable :: stress_z(:),stress_z_1(:)
    real(8), allocatable :: alp_pp(:)
    real(8), allocatable :: a_pot(:),local_force(:)
    integer , allocatable :: ipvt(:)
    real(8) En, local_energy,field,cross_field, sum_field, sum_field_1,a_f_i, poti
    real(8), allocatable :: dx(:), dy(:), dz(:), dr_sq(:)
    real(8), allocatable :: a_fi(:), e_zz(:),stress_33(:)
    real(8), allocatable :: force(:),force_1(:)
    integer itype
    real(8) pref,local_energy_pp
    real(8), save :: ew_beta, ew_gamma, CC_alpha, CC_beta, CC_gamma
    real(8) sum_cross_field_pp,sum_field_pp,sum_stress_field_pp
    real(8) sum_erf_field_pp
    logical l_i,l_j
    real(8) f_field, f_field_1, sum_force_field, sum_force_field_1,sum_stress_field_1
    real(8) vct4(4),CC_2,CC_22,CC_3,di_zz,exp_x2
    real(8), allocatable :: stress_pp(:) 
    real(8), allocatable :: local_force_pg(:),local_force_gp(:)
    real(8), allocatable :: buffer(:)
   
    allocate(field_grid(Ngrid),field_grid_1(Ngrid))
    allocate(field_force(Ngrid),field_force_1(Ngrid))
    allocate(stress_field(Ngrid),stress_field_1(Ngrid))
    allocate(MAT(1:Ngrid,1:Ngrid),alp(1:Ngrid),alp_1(1:Ngrid))
    allocate(a_pot(Natoms),local_force(Natoms))
    allocate(dx(MX_excluded),dy(MX_excluded),dz(MX_excluded),dr_sq(MX_excluded))
    allocate(a_fi(Natoms),e_zz(Natoms))
    allocate(alp_pp(1:Ngrid))
    allocate(force(1:Ngrid),force_1(Ngrid))
    allocate(stress_z(1:Ngrid),stress_z_1(1:Ngrid))
    allocate(buffer(Natoms))

  
    N = Ngrid - 1     
    CC_alpha = sqrt_Pi/Ewald_alpha
    CC_22 = 2.0d0*Ewald_alpha*sqrt_Pi
    CC_2 = CC_22 * 2.0d0
    CC_3 = 8.0d0*sqrt_Pi*Ewald_alpha**3
    i_Area = 1.0d0/area_xy   
    do k = 1, Ngrid
     sum_field=0.0d0
     sum_field_1=0.0d0;
     sum_force_field = 0.0d0
     sum_force_field_1=0.0d0
     sum_stress_field=0.0d0
     sum_stress_field_1=0.0d0
     do i = 1, Natoms 
        itype = i_type_atom(i)
!        zk = zzz(i)-z_grid(k)
         zk = z_grid(k) - zzz(i)
          qi = all_charges(i)
          di_zz = all_dipoles_zz(i)
          xxx = Ewald_alpha*zk
          erf_field = derf(xxx)
          exp_x2 = dexp(-xxx*xxx)
          field = qi*(CC_alpha*exp_x2+zk*(Pi*erf_field)) - di_zz*(Pi*erf_field)
          field_1 =   qi*(Pi*erf_field) - di_zz*exp_x2*CC_22  ! check it out.
          sum_field = sum_field + field !
          sum_field_1  = sum_field_1 + field_1
          f_field  = -Pi2*qi*erf_field + CC_2*di_zz*exp_x2
          f_field_1= (-CC_2*qi - CC_3 * zk * di_zz ) * exp_x2
          sum_force_field  = sum_force_field + f_field 
          sum_force_field_1= sum_force_field_1 + f_field_1
          sum_stress_field = sum_stress_field + zk * f_field
          sum_stress_field_1=sum_stress_field_1 + zk * f_field_1 
     enddo
    field_grid(k) = sum_field
    field_grid_1(k) = sum_field_1
    field_force(k) = sum_force_field
    field_force_1(k) = sum_force_field_1
    stress_field(k) = sum_stress_field
    stress_field_1(k) = sum_stress_field_1 
    enddo 

    field_grid = field_grid * ( (-2.0d0) * i_area)
    field_grid_1 = field_grid_1 * ( (-2.0d0) * i_area)
    field_force = field_force * (i_Area)
    field_force_1 = field_force_1* (i_Area)
    stress_field = stress_field*i_Area
    stress_field_1 = stress_field_1*i_Area 
   kk2 = mod(order,2)+1
   do i = 0,n
      x = z_grid(i+1)
      call deboor_cox(order,Ngrid, order+1, kkk, qq, x, bv)
      MAT(i+1,1:n+1) = bv(1:n+1,kk2)
   enddo ! i=1,n
    call invmat(MAT,Ngrid,Ngrid)
    do i = 1, Ngrid
      alp(i) = dot_product(MAT(i,:),(field_grid(:)))   !dot_product(MAT(i,:),B0(:))
      alp_1(i)=dot_product(MAT(i,:),(field_grid_1(:)))
    enddo
   do i = 1, Ngrid
     force(i) = dot_product(MAT(i,:),field_force(:))
     force_1(i) = dot_product(MAT(i,:),field_force_1(:))
   enddo

!!\?
   do i = 1, Ngrid
     stress_z(i) = dot_product(MAT(i,:),stress_field(:))
     stress_z_1(i) = dot_product(MAT(i,:),stress_field_1(:))
   enddo
!\\\?
    om2 = mod(order,2)+1
    local_stress_zz=0.0d0
    local_energy=0.0d0
    local_energy_pp=0.0d0
    local_force=0.0d0
    do i = 1, Natoms
       x = zzz(i)
       call deboor_cox(order,Ngrid, order+1, kkk, qq, x, bv)
       j   = kkk - order;
       field = dot_product(alp(j+1:kkk+1),bv(j+1:kkk+1,om2))
       field_1 = dot_product(alp_1(j+1:kkk+1),bv(j+1:kkk+1,om2))
       En = all_charges(i) * field + all_dipoles_zz(i)*field_1
       local_energy = local_energy + En;  a_pot(i) =  En; a_fi(i) =  field

       ff = - dot_product(force(j+1:kkk+1),bv(j+1:kkk+1,om2))
       ff1 = -dot_product(force_1(j+1:kkk+1),bv(j+1:kkk+1,om2))
       local_force(i) = local_force(i) + all_charges(i)*ff + all_dipoles_zz(i)*ff1

       szz = -dot_product(stress_z(j+1:kkk+1),bv(j+1:kkk+1,om2)) 
       szz_1 = -dot_product(stress_z_1(j+1:kkk+1),bv(j+1:kkk+1,om2)) 
       local_stress_zz = local_stress_zz + szz* all_charges(i) + szz_1 * all_dipoles_zz(i)

    enddo   

local_energy = local_energy * 0.50d0   !local_energy_pp + local_energy_gg + local_energy_pg
local_stress_zz = local_stress_zz * 0.5d0

En_Q_k0_cmplx = En_Q_k0_cmplx + local_energy
En_Q_cmplx = En_Q_cmplx + local_energy
En_Q = En_Q + local_energy
fzz(:) = fzz(:) + local_force(:)
stress_Qcmplx_k_eq_0(1:2)=local_energy
stress_Qcmplx_k_eq_0(3)  =local_stress_zz
stress_Qcmplx_k_eq_0(4)  = (local_energy+local_energy+local_stress_zz)/3.0d0
stress_Qcmplx_k_eq_0(5:10)=0.0d0
stress = stress + stress_Qcmplx_k_eq_0

if (l_need_2nd_profile) then
 atom_profile%pot = atom_profile%pot + a_pot
 atom_profile%Qpot = atom_profile%Qpot + a_pot
 atom_profile%fi = atom_profile%fi + a_fi
! do i = 1, Natoms; atom_profile(i)%buffer3(3) = atom_profile(i)%buffer3(3) + buffer(i) ; enddo
! atom_profile%szz = atom_profile%szz + stress_33
! atom_profile%EE_zz = atom_profile%EE_zz + e_zz
endif


    deallocate(field_grid,field_grid_1)
    deallocate(field_force,field_force_1)
    deallocate(stress_field,stress_field_1)
    deallocate(MAT,alp,alp_1)
    deallocate(a_pot,local_force)
    deallocate(dx,dy,dz,dr_sq)
    deallocate(a_fi,e_zz)
    deallocate(alp_pp)
    deallocate(force,force_1)
    deallocate(stress_z,stress_z_1)
    deallocate(buffer)

 end subroutine set_energy_2D_k0_Q_DIP

 

end module ewald_2D_k0_Q_DIP_module
