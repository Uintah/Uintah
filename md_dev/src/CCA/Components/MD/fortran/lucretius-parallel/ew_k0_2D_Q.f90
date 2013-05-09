module ewald_2D_k0_Q_module
 implicit none
 real(8), private, allocatable , save :: qq(:), bv(:,:),z_grid(:),db(:,:)
 integer, private :: order, Ngrid, N_size_qq
 private :: set_energy_2D_k0_Q
 public :: driver_ewald_2D_k0_Q
 
 contains

 subroutine driver_ewald_2D_k0_Q
! 1G means that all gaussian charges have the same width parameter 
! that is an extrodinary simplification of the problem
   use Ewald_data ! order_spline_zz_k0, n_grid_zz_k0
   use spline_z_k0_module
   use energies_data ! for debug
   implicit none
   call initialize
   call set_energy_2D_k0_Q
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
 end subroutine driver_ewald_2D_k0_Q
 
! include 'Ew_2D_k0_Q_SLOW.f90' ! Its dependency it is now in Makefile

 subroutine set_energy_2D_k0_Q
    use spline_z_k0_module
    use array_math
    use sim_cel_data
    use ALL_atoms_data, only : xx,yy,zz,all_charges,Natoms, fzz, &
                               is_charge_distributed, i_type_atom, zzz
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
    integer i,j,k,n,kk2,om2,kkk,i1
    integer jlim1,jlim2
    real(8) i_Area,CC1,zk,x,qi,xxx, ff,erf_field, sum_erf_field,field1,zij,qij,qj
    real(8) sum_stress_field, local_stress_zz
    real(8), allocatable :: MAT(:,:), alp_P(:),field_grid_pp(:)
    real(8), allocatable :: erf_field_pp(:), stress_field_pp(:)
    real(8), allocatable :: alp_pp(:)
    real(8), allocatable :: a_pot(:),local_force(:)
    integer , allocatable :: ipvt(:)
    real(8) En, local_energy,field,cross_field, sum_field, a_f_i, poti
    real(8), allocatable :: dx(:), dy(:), dz(:), dr_sq(:)
    real(8), allocatable :: a_fi(:), e_zz(:),stress_33(:)
    real(8), allocatable :: field_ERF_grid_pp(:), force_pp(:)
    integer itype
    real(8) pref,local_energy_pp
    real(8), save :: ew_beta, ew_gamma, CC_alpha, CC_beta, CC_gamma
    real(8) sum_cross_field_pp,sum_field_pp,sum_stress_field_pp
    real(8) sum_erf_field_pp
    logical l_i,l_j
    real(8) vct4(4),szz
    real(8), allocatable :: stress_pp(:) 
    real(8), allocatable :: local_force_pg(:),local_force_gp(:)
   
    allocate(field_grid_pp(Ngrid),stress_field_pp(Ngrid))
    allocate(MAT(1:Ngrid,1:Ngrid),alp_P(1:Ngrid))
    allocate(a_pot(Natoms),local_force(Natoms))
    allocate(erf_field_pp(natoms))
    allocate(dx(MX_excluded),dy(MX_excluded),dz(MX_excluded),dr_sq(MX_excluded))
    allocate(a_fi(Natoms),e_zz(Natoms),stress_33(Natoms))
    allocate(alp_pp(1:Ngrid))
    allocate(field_ERF_grid_pp(1:Ngrid))
    allocate(force_pp(1:Ngrid))
    allocate(stress_pp(1:Ngrid))
  
    N = Ngrid - 1     
    CC_alpha = sqrt_Pi/Ewald_alpha
    i_Area = 1.0d0/area_xy   
    do k = 1, Ngrid
     sum_field=0.0d0
     sum_erf_field=0.0d0
     sum_stress_field=0.0d0
     sum_field_pp=0.0d0; 
     sum_stress_field_pp=0.0d0
     sum_erf_field_pp = 0.0d0
     do i = 1, Natoms 
        itype = i_type_atom(i)
        zk = zzz(i)-z_grid(k)
          qi = all_charges(i)
          xxx = Ewald_alpha*zk
          field = qi*(CC_alpha*dexp(-xxx*xxx)+zk*Pi*derf(xxx))
          erf_field = derf(xxx)
          sum_field_pp = sum_field_pp + field !
          sum_erf_field_pp = sum_erf_field_pp + qi*erf_field
          sum_stress_field_pp = sum_stress_field_pp + erf_field*zk*qi
     enddo
    field_grid_pp(k) = sum_field_pp
    stress_field_pp(k) = sum_stress_field_pp
    field_ERF_grid_pp(k) = sum_erf_field_pp
    enddo 

    field_grid_pp = field_grid_pp * ( (-2.0d0) * i_area)
    field_ERF_grid_pp = field_ERF_grid_pp * (Pi*2.0d0*i_area)
    stress_field_pp = stress_field_pp * Pi* ( (-2.0d0) * i_area)

    
   kk2 = mod(order,2)+1
   do i = 0,n
      x = z_grid(i+1)
      call deboor_cox(order,Ngrid, order+1, kkk, qq, x, bv)
      MAT(i+1,1:n+1) = bv(1:n+1,kk2)
   enddo ! i=1,n
    call invmat(MAT,Ngrid,Ngrid)
    do i = 1, Ngrid
      alp_P(i) = dot_product(MAT(i,:),(field_grid_pp(:)))   !dot_product(MAT(i,:),B0(:))
    enddo
   do i = 1, Ngrid
     force_pp(i) = dot_product(MAT(i,:),field_ERF_grid_pp(:))
   enddo
   do i = 1, Ngrid
     stress_pp(i) = dot_product(MAT(i,:),stress_field_pp(:))
   enddo

    om2 = mod(order,2)+1
    local_stress_zz=0.0d0
    local_energy=0.0d0
    local_energy_pp=0.0d0
    local_force=0.0d0
    do i = 1, Natoms
       x = zzz(i)
       call deboor_cox(order,Ngrid, order+1, kkk, qq, x, bv)
       j   = kkk - order;
       field = dot_product(alp_P(j+1:kkk+1),bv(j+1:kkk+1,om2))
       En = all_charges(i) * field
       a_fi(i) =  field
       local_energy = local_energy + En
       a_pot(i) =  En
!!!         ff = spline_k0_deriv(order, Ngrid, kkk, x, alp_P, qq,db)
!!!         ff = -ff
       ff = -dot_product(force_pp(j+1:kkk+1),bv(j+1:kkk+1,om2))
       local_force(i) = local_force(i) + all_charges(i)*ff
       e_zz(i) =  ff
       szz = -dot_product(stress_pp(j+1:kkk+1),bv(j+1:kkk+1,om2)) * all_charges(i)
       local_stress_zz = local_stress_zz + szz 
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
! atom_profile%szz = atom_profile%szz + stress_33
! atom_profile%EE_zz = atom_profile%EE_zz + e_zz
endif

    deallocate(field_grid_pp,stress_field_pp)
    deallocate(MAT,alp_P)
    deallocate(a_pot,local_force)
    deallocate(erf_field_pp)
    deallocate(dx,dy,dz,dr_sq)
    deallocate(a_fi,e_zz,stress_33)
    deallocate(alp_pp)
    deallocate(field_ERF_grid_pp)
    deallocate(force_pp)
    deallocate(stress_pp)
 end subroutine set_energy_2D_k0_Q

 

end module ewald_2D_k0_Q_module
