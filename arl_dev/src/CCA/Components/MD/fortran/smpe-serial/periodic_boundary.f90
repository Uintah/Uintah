module boundaries
contains

 subroutine periodic_images(xx,yy,zz)
      use sim_cel_data
      implicit none
      real(8), intent(INOUT) :: xx(:),yy(:),zz(:)  ! r (N,3)
      real(8) i_sim_cel(9), isim
     
      select case (i_boundary_CTRL)
      case(0) ! VACUUM  do nothing
      case(1) ! SLAB (periodicity in 2 directions and aperiodic in OZ)
        i_sim_cel(1) = 1.0d0/sim_cel(1)   ;   i_sim_cel(5) = 1.0d0/sim_cel(5)
        xx(:) = xx(:) - sim_cel(1)*(dble(INT(2.0d0*(i_sim_cel(1)*xx(:)))) -dble(INT((i_sim_cel(1)*xx(:)))) )
        yy(:) = yy(:) - sim_cel(5)*(dble(INT(2.0d0*(i_sim_cel(5)*yy(:)))) -dble(INT((i_sim_cel(5)*yy(:)))) )
!        zz(:) = zz(:)
      case(2,3) ! Paralepipedic
        i_sim_cel(1) = 1.0d0/sim_cel(1)   ;   i_sim_cel(5) = 1.0d0/sim_cel(5)   ; i_sim_cel(9) = 1.0d0/sim_cel(9)
        xx(:) = xx(:) - sim_cel(1)*(dble(INT(2.0d0*(i_sim_cel(1)*xx(:)))) -dble(INT((i_sim_cel(1)*xx(:)))) )
        yy(:) = yy(:) - sim_cel(5)*(dble(INT(2.0d0*(i_sim_cel(5)*yy(:)))) -dble(INT((i_sim_cel(5)*yy(:)))) )
        zz(:) = zz(:) - sim_cel(9)*(dble(INT(2.0d0*(i_sim_cel(9)*zz(:)))) -dble(INT((i_sim_cel(9)*zz(:)))) )
      case(4) ! ortorombic
        !  
      case default
        print*, 'NOT defined case in periodic_images',i_boundary_CTRL
STOP
      end select

 end subroutine periodic_images


 subroutine get_reciprocal_cut
  use cut_off_data, only : reciprocal_cut,reciprocal_cut_sq
  use sim_cel_data, only : Reciprocal_cel, i_boundary_CTRL
  use Ewald_data 
  implicit none
  real(8), parameter :: amplify_a_bit = 1.10d0
  if (i_boundary_CTRL /= 1) then  ! 3D
    if (i_type_EWALD_CTRL==2) then 
      reciprocal_cut = min(dble(k_max_x)*Reciprocal_cel(1),dble(k_max_y)*Reciprocal_cel(5),dble(k_max_z)*Reciprocal_cel(9))
      reciprocal_cut = 0.5d0*reciprocal_cut*amplify_a_bit
      reciprocal_cut_sq = reciprocal_cut**2
    else if (i_type_EWALD_CTRL==1) then
      reciprocal_cut = min(dble(nfftx)*Reciprocal_cel(1),dble(nffty)*Reciprocal_cel(5),dble(nfftz)*Reciprocal_cel(9))
      reciprocal_cut = 0.5d0*reciprocal_cut*amplify_a_bit
      reciprocal_cut_sq = reciprocal_cut**2

  endif
  else   ! i_boundary_CTRL ==1  ! 2D
    if (i_type_EWALD_CTRL==2) then
      reciprocal_cut = min(dble(k_max_x)*Reciprocal_cel(1),dble(k_max_y)*Reciprocal_cel(5),dble(k_max_z)*Reciprocal_cel(9))
      reciprocal_cut = 0.5d0*reciprocal_cut*amplify_a_bit
      reciprocal_cut_sq = reciprocal_cut**2
    else if (i_type_EWALD_CTRL==1) then
      reciprocal_cut = min(dble(nfftx)*Reciprocal_cel(1),dble(nffty)*Reciprocal_cel(5),dble(nfftz)*Reciprocal_cel(9))
! shouldn't last term be just Pi2? (replace : dble(nfftz)*Reciprocal_cel(9) with Pi2 )????? Double check that...
      reciprocal_cut = 0.5d0*reciprocal_cut*amplify_a_bit
      reciprocal_cut_sq = reciprocal_cut**2
    endif
  endif

 end subroutine get_reciprocal_cut


 subroutine get_reduced_coordinates
   use sim_cel_data
   use ALL_atoms_data, only : xxx,yyy,zzz,ttx,tty,ttz,Natoms
   implicit none
   real(8) i_sim_cel(9)
  
    i_sim_cel(1) = 1.0d0/sim_cel(1)   ;   i_sim_cel(5) = 1.0d0/sim_cel(5)   ; i_sim_cel(9) = 1.0d0/sim_cel(9)
    ttx(:) = xxx(:) - sim_cel(1)*(dble(INT(2.0d0*(i_sim_cel(1)*xxx(:)))) -dble(INT((i_sim_cel(1)*xxx(:)))) )
    tty(:) = yyy(:) - sim_cel(5)*(dble(INT(2.0d0*(i_sim_cel(5)*yyy(:)))) -dble(INT((i_sim_cel(5)*yyy(:)))) )
    ttz(:) = zzz(:) - sim_cel(9)*(dble(INT(2.0d0*(i_sim_cel(9)*zzz(:)))) -dble(INT((i_sim_cel(9)*zzz(:)))) )
    ttx(:) = ttx(:) * i_sim_cel(1)
    tty(:) = tty(:) * i_sim_cel(5)
    ttz(:) = ttz(:) * i_sim_cel(9)
 
      
 end subroutine get_reduced_coordinates


 subroutine cel_properties(l_reduced_details)
      use math, only : invert3, get_minors_3x3
      use math_constants, only : Pi2,Pi4
      use sim_cel_data
      implicit none
      logical, intent(IN) :: l_reduced_details ! if evaluate details of reciprocal cel 
      real(8) det,a11,a12,a13,a21,a22,a23,a31,a32,a33
      call invert3(sim_cel,inverse_cel,det)
      cel_a2 = dsqrt(dot_product(sim_cel(1:3),sim_cel(1:3))) 
      cel_b2 = dsqrt(dot_product(sim_cel(4:6),sim_cel(4:6)))
      cel_c2 = dsqrt(dot_product(sim_cel(7:9),sim_cel(7:9)))
      cel_cos_ab=(sim_cel(1)*sim_cel(4)+sim_cel(2)*sim_cel(5)+sim_cel(3)*sim_cel(6))/(cel_a2*cel_b2)
      cel_cos_ac=(sim_cel(1)*sim_cel(7)+sim_cel(2)*sim_cel(8)+sim_cel(3)*sim_cel(9))/(cel_a2*cel_c2)
      cel_cos_bc=(sim_cel(4)*sim_cel(7)+sim_cel(5)*sim_cel(8)+sim_cel(6)*sim_cel(9))/(cel_b2*cel_c2)
      call get_minors_3x3(sim_cel,a11,a12,a13,a21,a22,a23,a31,a32,a33)
      Volume = dabs(sim_cel(1)*a11+sim_cel(2)*a12+sim_cel(3)*a13)
      Area_xy = cel_a2*cel_b2*dsqrt(1.0d0-cel_cos_ab**2)
      cel_cos_a_bxc = Volume / dsqrt(a11*a11+a12*a12+a13*a13) ! 
      cel_cos_b_axc = Volume / dsqrt(a21*a21+a22*a22+a23*a23)
      cel_cos_c_axb = Volume / dsqrt(a31*a31+a32*a32+a33*a33)    
      Reciprocal_cel(1:9) = Pi2*inverse_cel(1:9)
      Reciprocal_Volume = Pi4/Volume
    if (l_reduced_details) then
      inv_cel_a2 = dsqrt(dot_product(inverse_cel(1:3),inverse_cel(1:3)))
      inv_cel_b2 = dsqrt(dot_product(inverse_cel(4:6),inverse_cel(4:6)))
      inv_cel_c2 = dsqrt(dot_product(inverse_cel(7:9),inverse_cel(7:9)))
      inv_cel_cos_ab=(inverse_cel(1)*inverse_cel(4)+inverse_cel(2)*inverse_cel(5)+inverse_cel(3)*inverse_cel(6))&
                     /(inv_cel_a2*inv_cel_b2)
      inv_cel_cos_ac=(inverse_cel(1)*inverse_cel(7)+inverse_cel(2)*inverse_cel(8)+inverse_cel(3)*inverse_cel(9))&
                     /(inv_cel_a2*inv_cel_c2)
      inv_cel_cos_bc=(inverse_cel(4)*inverse_cel(7)+inverse_cel(5)*inverse_cel(8)+inverse_cel(6)*inverse_cel(9))&
                     /(inv_cel_b2*inv_cel_c2)
      call get_minors_3x3(inverse_cel,a11,a12,a13,a21,a22,a23,a31,a32,a33)
      inv_Volume = dabs(inverse_cel(1)*a11+inverse_cel(2)*a12+inverse_cel(3)*a13)
      inv_cel_cos_a_bxc = inv_Volume/dsqrt(a11*a11+a12*a12+a13*a13) 
      inv_cel_cos_b_axc = inv_Volume/dsqrt(a21*a21+a22*a22+a23*a23)
      inv_cel_cos_c_axb = inv_Volume/dsqrt(a31*a31+a32*a32+a33*a33)
      Reciprocal_perp(1) = Pi2*inv_cel_cos_a_bxc
      Reciprocal_perp(2) = Pi2*inv_cel_cos_b_axc
      Reciprocal_perp(3) = Pi2*inv_cel_cos_c_axb
    endif

 end subroutine cel_properties

end module boundaries
