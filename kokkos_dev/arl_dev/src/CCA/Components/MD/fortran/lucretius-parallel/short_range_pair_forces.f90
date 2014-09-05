 module short_range_pair_forces
 implicit none

! include 'variables_short_pairs.var'

 private :: finalize_scalar_props 
 
 contains

 include 'pair_short_forces_vdw.f90' 
 include 'Q_2_forces.f90'
 include 'Q_2_forces_dipoles.f90'
 include 'pair_short_forces_Q.f90'

 include 'nonbonded_vdw_2_forces.f90'



! include 'Q_2_forces_qGi.f90' ! gauss charge i
! include 'Q_2_forces_dipoles.f90' ! for dipoles If neither p-charge or g-charge are p



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 subroutine finalize_scalar_props
 use energies_data
 use stresses_data
 use variables_short_pairs
 implicit none
 integer i,j
 real(8) t(9),a,b
 t=0.0d0
 En_Q = En_Q + en_Qreal
!print*, 'stress vdw =',stress_vdw_xx,stress_vdw_xy,stress_vdw_zz
!print*, 'stress vdw =',stress_vdw_xy,stress_vdw_xz,stress_vdw_yz
!print*, 'stress Q=',stress_xx,stress_yy,stress_zz
!print*, 'stress Q=',stress_xy,stress_xz,stress_yz
!stop

 stress_Qreal(1) = stress_xx
 stress_Qreal(2) = stress_yy
 stress_Qreal(3) = stress_zz
 stress_Qreal(4) = (stress_xx+stress_yy+stress_zz)/3.0d0
 stress_Qreal(5) = stress_xy
 stress_Qreal(6) = stress_xz
 stress_Qreal(7) = stress_yz
 stress_Qreal(8) = stress_xy
 stress_Qreal(9) = stress_xz
 stress_Qreal(10) = stress_yz


 stress_vdw(1) = stress_vdw_xx 
 stress_vdw(2) = stress_vdw_yy 
 stress_vdw(3) = stress_vdw_zz 
 stress_vdw(4) = (stress_vdw_xx+stress_vdw_yy+stress_vdw_zz)/3.0d0
 stress_vdw(5) = stress_vdw_xy 
 stress_vdw(6) = stress_vdw_xz 
 stress_vdw(7) = stress_vdw_yz 

 stress_vdw(8) = stress_vdw_xy 
 stress_vdw(9) = stress_vdw_xz 
 stress_vdw(10) = stress_vdw_yz 

 stress(:) = stress(:) + stress_Qreal(:) + stress_vdw(:)

!print*, 'after eval stress=',stress(1:10)
 end subroutine finalize_scalar_props

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! include 'nonbonded_vdw_2_forces.f90'
! include 'Q_2_forces_qpi.f90' ! point charge i
! include 'Q_2_forces_qGi.f90' ! gauss charge i
! include 'Q_2_forces_dipoles.f90' ! for dipoles If neither p-charge or g-charge are present do not eval the field on that atom for now.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 end module short_range_pair_forces
