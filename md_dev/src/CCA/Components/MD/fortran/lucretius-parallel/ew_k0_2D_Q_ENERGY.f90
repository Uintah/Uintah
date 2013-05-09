module ewald_2D_k0_Q_module_ENERGY
 implicit none
 real(8), private, allocatable , save :: qq(:), bv(:,:),z_grid(:),db(:,:)
 integer, private :: order, Ngrid, N_size_qq
 private :: set_energy_2D_k0_Q_ENERGY
 public :: driver_ewald_2D_k0_Q_ENERGY
   
 contains

 subroutine driver_ewald_2D_k0_Q_ENERGY
! 1G means that all gaussian charges have the same width parameter 
! that is an extrodinary simplification of the problem
   use Ewald_data ! order_spline_zz_k0, n_grid_zz_k0
   use spline_z_k0_module
   use energies_data ! for debug
   implicit none
   call initialize
   call set_energy_2D_k0_Q_ENERGY
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
 end subroutine driver_ewald_2D_k0_Q_ENERGY
 
! include 'Ew_2D_k0_Q_SLOW_ENERGY.f90' ! its dependency is now in Makefile

 subroutine set_energy_2D_k0_Q_ENERGY
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
    real(8) i_Area,CC1,zk,x,qi,xxx, zij,qij,qj
    real(8), allocatable :: MAT(:,:), alp_P(:),field_grid_pp(:)
    real(8), allocatable :: alp_pp(:)
    integer , allocatable :: ipvt(:)
    real(8) En, local_energy,field,cross_field, sum_field, a_f_i, poti
    real(8), allocatable :: dx(:), dy(:), dz(:), dr_sq(:)
    integer itype
    real(8) pref,local_energy_pp
    real(8), save :: ew_beta, ew_gamma, CC_alpha, CC_beta, CC_gamma
    real(8) sum_field_pp
    logical l_i,l_j
    real(8) vct4(4),szz
   
    allocate(field_grid_pp(Ngrid))
    allocate(MAT(1:Ngrid,1:Ngrid),alp_P(1:Ngrid))
    allocate(dx(MX_excluded),dy(MX_excluded),dz(MX_excluded),dr_sq(MX_excluded))
    allocate(alp_pp(1:Ngrid))
  
    N = Ngrid - 1     
    CC_alpha = sqrt_Pi/Ewald_alpha
    i_Area = 1.0d0/area_xy   
    do k = 1, Ngrid
     sum_field_pp=0.0d0; 
     do i = 1, Natoms 
        itype = i_type_atom(i)
        zk = zzz(i)-z_grid(k)
          qi = all_charges(i)
          xxx = Ewald_alpha*zk
          field = qi*(CC_alpha*dexp(-xxx*xxx)+zk*Pi*derf(xxx))
          sum_field_pp = sum_field_pp + field !
     enddo
    field_grid_pp(k) = sum_field_pp
    enddo 

    field_grid_pp = field_grid_pp * ( (-2.0d0) * i_area)
    
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
    om2 = mod(order,2)+1
    local_energy=0.0d0
    do i = 1, Natoms
       x = zzz(i)
       call deboor_cox(order,Ngrid, order+1, kkk, qq, x, bv)
       j   = kkk - order;
       field = dot_product(alp_P(j+1:kkk+1),bv(j+1:kkk+1,om2))
       En = all_charges(i) * field
       local_energy = local_energy + En
    enddo   

local_energy = local_energy * 0.50d0   !local_energy_pp + local_energy_gg + local_energy_pg

En_Q_k0_cmplx = En_Q_k0_cmplx + local_energy
En_Q_cmplx = En_Q_cmplx + local_energy
En_Q = En_Q + local_energy


    deallocate(field_grid_pp)
    deallocate(MAT,alp_P)
    deallocate(dx,dy,dz,dr_sq)
    deallocate(alp_pp)
 end subroutine set_energy_2D_k0_Q_ENERGY

 

end module ewald_2D_k0_Q_module_ENERGY
