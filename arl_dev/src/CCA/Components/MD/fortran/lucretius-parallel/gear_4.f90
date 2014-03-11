
 module gear_4_module_MOL
 implicit none

 private :: ber_G4_thermo_predict
 private :: ber_G4_thermo_correct
 private :: ber_G4_thermo_adjust
 private :: ber_G4_BARR_XYZ_adjust
 private :: predictor_4th_order_gear
 private :: corrector_4th_order_gear
 public :: NVE_G4_MOL
 public :: NVT_G4_BER_MOL
 public :: NPT_G4_BER_XYZ_MOL

 contains

 subroutine NVE_G4_MOL(step)
 use kinetics, only : mol_kinetics
 use convert_mol_to_atom
 use rotor_utils, only : torques_in_body_frame
 implicit none
 integer, intent(IN) :: step
 if (step == 1) then
   call predictor_4th_order_gear
 else if (step == 2) then
   call from_atom_to_mol
   call torques_in_body_frame
   call corrector_4th_order_gear
   call mol_kinetics
 else
   print*, 'case defalut in gear_4%NVE_G4 for step; STOP'
   STOP
 endif
 end subroutine NVE_G4_MOL 

 subroutine NVT_G4_BER_MOL(step)
 use kinetics, only : mol_kinetics
 use convert_mol_to_atom
 use rotor_utils, only : torques_in_body_frame
 use sim_cel_data
 implicit none
 integer, intent(IN) :: step
 if (step == 1) then
   call predictor_4th_order_gear
 else if (step == 2) then
   call from_atom_to_mol
   call torques_in_body_frame
   call ber_G4_thermo_adjust
   call corrector_4th_order_gear
   call mol_kinetics
 else
   print*, 'case defalut in gear_4%NVT_G4_BER_MOL for step; STOP'
   STOP
 endif
 end subroutine NVT_G4_BER_MOL

 subroutine NPT_G4_BER_XYZ_MOL(step)
 use kinetics, only : mol_kinetics
 use convert_mol_to_atom
 use rotor_utils, only : torques_in_body_frame
 implicit none
 integer, intent(IN) :: step
 if (step == 1) then
   call predictor_4th_order_gear
 else if (step == 2) then
   call from_atom_to_mol
   call torques_in_body_frame
   call ber_G4_thermo_adjust
   call ber_G4_BARR_XYZ_adjust
   call corrector_4th_order_gear
   call mol_kinetics
 else
   print*, 'case defalut in gear_4%NPT_G4_BER_XYZ_MOL for step; STOP'
   STOP
 endif

 end subroutine NPT_G4_BER_XYZ_MOL


  SUBROUTINE predictor_4th_order_gear
  use qn_and_hi_deriv_data
  use qn_and_low_deriv_data 
  use ALL_rigid_mols_data
  use ALL_mols_data, only : Nmols, l_WALL_MOL_CTRL,mol_xyz
  use rotor_utils, only : renormalize_quaternions, atom_in_lab_frame, get_mol_orient_atom_xyz
  use kinetics, only : get_instant_MOL_temperature
  use ensamble_data, only : Temperature_trans_Calc, Temperature_rot_Calc
  implicit none
  integer i,j

   do i = 1, Nmols
   if (.not.l_WALL_MOL_CTRL(i)) then
    mol_xyz(i,:)=mol_xyz(i,:)+mol_xyz_1_deriv(i,:)+mol_xyz_2_deriv(i,:)+mol_xyz_3_deriv(i,:)+mol_xyz_4_deriv(i,:)
    mol_xyz_1_deriv(i,:)=mol_xyz_1_deriv(i,:)+2.0d0*mol_xyz_2_deriv(i,:)+3.0d0*mol_xyz_3_deriv(i,:)+4.0d0*mol_xyz_4_deriv(i,:)
    mol_xyz_2_deriv(i,:)=mol_xyz_2_deriv(i,:)+3.0d0*mol_xyz_3_deriv(i,:)+6.0d0*mol_xyz_4_deriv(i,:)
    mol_xyz_3_deriv(i,:)=mol_xyz_3_deriv(i,:)+4.0d0*mol_xyz_4_deriv(i,:)

    mol_MOM(i,:)=mol_MOM(i,:)+mol_MOM_1_deriv(i,:)+mol_MOM_2_deriv(i,:)+mol_MOM_3_deriv(i,:)+mol_MOM_4_deriv(i,:)
    mol_MOM_1_deriv(i,:)=mol_MOM_1_deriv(i,:)+2.0d0*mol_MOM_2_deriv(i,:)+3.0d0*mol_MOM_3_deriv(i,:)+4.0d0*mol_MOM_4_deriv(i,:)
    mol_MOM_2_deriv(i,:)=mol_MOM_2_deriv(i,:)+3.0d0*mol_MOM_3_deriv(i,:)+6.0d0*mol_MOM_4_deriv(i,:)
    mol_MOM_3_deriv(i,:)=mol_MOM_3_deriv(i,:)+4.0d0*mol_MOM_4_deriv(i,:)

    mol_ANG(i,:)=mol_ANG(i,:)+mol_ANG_1_deriv(i,:)+mol_ANG_2_deriv(i,:)+mol_ANG_3_deriv(i,:)+mol_ANG_4_deriv(i,:)
    mol_ANG_1_deriv(i,:)=mol_ANG_1_deriv(i,:)+2.0d0*mol_ANG_2_deriv(i,:)+3.0d0*mol_ANG_3_deriv(i,:)+4.0d0*mol_ANG_4_deriv(i,:)
    mol_ANG_2_deriv(i,:)=mol_ANG_2_deriv(i,:)+3.0d0*mol_ANG_3_deriv(i,:)+6.0d0*mol_ANG_4_deriv(i,:)
    mol_ANG_3_deriv(i,:)=mol_ANG_3_deriv(i,:)+4.0d0*mol_ANG_4_deriv(i,:)
    qn(i,:)=qn(i,:)+qn_1_deriv(i,:)+qn_2_deriv(i,:)+qn_3_deriv(i,:)+qn_4_deriv(i,:)
    call renormalize_quaternions
    qn_1_deriv(i,:)=qn_1_deriv(i,:)+2.0d0*qn_2_deriv(i,:)+3.0d0*qn_3_deriv(i,:)+4.0d0*qn_4_deriv(i,:)
    qn_2_deriv(i,:)=qn_2_deriv(i,:)+3.0d0*qn_3_deriv(i,:)+6.0d0*qn_4_deriv(i,:)
    qn_3_deriv(i,:)=qn_3_deriv(i,:)+4.0d0*qn_4_deriv(i,:)
   else
    mol_xyz_1_deriv(i,:)=0.0d0; mol_xyz_2_deriv(i,:)=0.0d0; mol_xyz_3_deriv(i,:)=0.0d0; mol_xyz_4_deriv(i,:)=0.0d0
    mol_MOM_1_deriv(i,:)=0.0d0; mol_MOM_2_deriv(i,:)=0.0d0; mol_MOM_3_deriv(i,:)=0.0d0; mol_MOM_4_deriv(i,:)=0.0d0
    mol_ANG_1_deriv(i,:)=0.0d0; mol_ANG_2_deriv(i,:)=0.0d0; mol_ANG_3_deriv(i,:)=0.0d0; mol_ANG_4_deriv(i,:)=0.0d0
    qn_1_deriv(i,:)=0.0d0; qn_2_deriv(i,:)=0.0d0; qn_3_deriv(i,:)=0.0d0; qn_4_deriv(i,:)=0.0d0
   endif
   enddo


   call get_mol_orient_atom_xyz
   call atom_in_lab_frame
   call get_instant_MOL_temperature(Temperature_trans_Calc, Temperature_rot_Calc)

   end SUBROUTINE predictor_4th_order_gear

   
  SUBROUTINE corrector_4th_order_gear
  use qn_and_low_deriv_data
  use qn_and_hi_deriv_data
  use ALL_rigid_mols_data
  use ALL_mols_data, only : Nmols, l_WALL_MOL_CTRL, mol_force,mol_xyz
  use integrate_data
  use rotor_utils, only : renormalize_quaternions, atom_in_lab_frame, torques_in_body_frame, get_mol_orient_atom_xyz
  use convert_mol_to_atom
  use kinetics, only : add_kinetic_pressure,get_mol_stresses,get_instant_MOL_temperature
  use ensamble_data, only : Temperature_trans_Calc, Temperature_rot_Calc
  implicit none
  integer i,i_type,j,k
  real(8), parameter :: F01=251.0d0/720.0d0 !put the actual value better
  real(8), parameter :: F21=11.0d0/12.0d0
  real(8), parameter :: F31=1.0d0/3.0d0
  real(8), parameter :: F41=1.0d0/24.0d0
  real(8) correct(3),qn_correct(4)

  do i = 1, Nmols
  if (.not.l_WALL_MOL_CTRL(i)) then 

   do k=1,3
    Correct(k)=mol_xyz_1_deriv(i,k)-time_step*Inverse_Molar_mass(i)*mol_MOM(i,k)
   enddo

    mol_xyz(i,:)=mol_xyz(i,:)-Correct(:)*F01
    mol_xyz_1_deriv(i,:)=mol_xyz_1_deriv(i,:)-Correct(:)
    mol_xyz_2_deriv(i,:)=mol_xyz_2_deriv(i,:)-F21*Correct(:)
    mol_xyz_3_deriv(i,:)=mol_xyz_3_deriv(i,:)-F31*Correct(:)
    mol_xyz_4_deriv(i,:)=mol_xyz_4_deriv(i,:)-F41*Correct(:)
 
    do k = 1, 3
     Correct(k)=mol_MOM_1_deriv(i,k)-time_step*mol_force(i,k)
    enddo 
    mol_MOM(i,:)=mol_MOM(i,:)-Correct(:)*F01
    mol_MOM_1_deriv(i,:)=mol_MOM_1_deriv(i,:)-Correct(:)
    mol_MOM_2_deriv(i,:)=mol_MOM_2_deriv(i,:)-F21*Correct(:)
    mol_MOM_3_deriv(i,:)=mol_MOM_3_deriv(i,:)-F31*Correct(:)
    mol_MOM_4_deriv(i,:)=mol_MOM_4_deriv(i,:)-F41*Correct(:)

    do j=1,3
      correct(j)=inverse_Inertia_MAIN(i,j)*mol_ANG(i,j) !ij=1,3
    enddo

    qn_correct(1)=qn_1_deriv(i,1)-(time_step*0.5d0)*(-qn(i,3)*correct(1)-qn(i,4)*correct(2)+qn(i,2)*correct(3))
    qn_correct(2)=qn_1_deriv(i,2)-(time_step*0.5d0)*(qn(i,4)*correct(1)-qn(i,3)*correct(2)-qn(i,1)*correct(3))
    qn_correct(3)=qn_1_deriv(i,3)-(time_step*0.5d0)*(qn(i,1)*correct(1)+qn(i,2)*correct(2)+qn(i,4)*correct(3))
    qn_correct(4)=qn_1_deriv(i,4)-(time_step*0.5d0)*(-qn(i,2)*correct(1)+qn(i,1)*correct(2)-qn(i,3)*correct(3))

    qn(i,:)=qn(i,:)-F01*qn_correct(:)
    qn_1_deriv(i,:)=qn_1_deriv(i,:)-qn_correct(:)
    qn_2_deriv(i,:)=qn_2_deriv(i,:)-qn_correct(:)*F21
    qn_3_deriv(i,:)=qn_3_deriv(i,:)-qn_correct(:)*F31
    qn_4_deriv(i,:)=qn_4_deriv(i,:)-qn_correct(:)*F41

    qn_Correct(1)=mol_ANG_1_deriv(i,1)-time_step*(mol_torque(i,1)+Inertia_SEC(i,3)*correct(2)*correct(3))
    qn_correct(2)=mol_ANG_1_deriv(i,2)-time_step*(mol_torque(i,2)+Inertia_SEC(i,2)*correct(1)*correct(3))
    qn_correct(3)=mol_ANG_1_deriv(i,3)-time_step*(mol_torque(i,3)+Inertia_SEC(i,1)*correct(1)*correct(2))

    mol_ANG(i,:)=mol_ANG(i,:)-F01*qn_correct(1:3)

    mol_ANG_1_deriv(i,:)=mol_ANG_1_deriv(i,:)-qn_correct(1:3)
    mol_ANG_2_deriv(i,:)=mol_ANG_2_deriv(i,:)-F21*qn_correct(1:3)
    mol_ANG_3_deriv(i,:)=mol_ANG_3_deriv(i,:)-F31*qn_correct(1:3)
    mol_ANG_4_deriv(i,:)=mol_ANG_4_deriv(i,:)-F41*qn_correct(1:3)

   endif
   enddo 


   call renormalize_quaternions

   where (.not.l_non_linear_rotor)
    mol_ANG(:,1)=0.0d0
    mol_ANG(:,2)=0.0d0
    mol_ANG(:,3)=0.0d0
    mol_ANG_1_deriv(:,1)=0.0d0
    mol_ANG_1_deriv(:,2)=0.0d0
    mol_ANG_1_deriv(:,3)=0.0d0
    mol_ANG_2_deriv(:,1)=0.0d0
    mol_ANG_2_deriv(:,2)=0.0d0
    mol_ANG_2_deriv(:,3)=0.0d0
    mol_ANG_3_deriv(:,1)=0.0d0
    mol_ANG_3_deriv(:,2)=0.0d0
    mol_ANG_3_deriv(:,3)=0.0d0
    mol_ANG_4_deriv(:,1)=0.0d0
    mol_ANG_4_deriv(:,2)=0.0d0
    mol_ANG_4_deriv(:,3)=0.0d0
   end where

  call get_mol_orient_atom_xyz
  call atom_in_lab_frame
  call get_mol_stresses ! update kinetic stresses
  call add_kinetic_pressure 
  call get_instant_MOL_temperature(Temperature_trans_Calc, Temperature_rot_Calc)

 end SUBROUTINE corrector_4th_order_gear

 subroutine ber_G4_thermo_predict
 end subroutine ber_G4_thermo_predict

 subroutine ber_G4_thermo_correct
 end subroutine ber_G4_thermo_correct

 subroutine ber_G4_thermo_adjust
    use ALL_mols_data, only : Nmols,l_WALL_MOL_CTRL, mol_force
    use ALL_rigid_mols_data, only : mol_ANG, mol_MOM, mol_torque
    use ensamble_data
    use kinetics, only : get_instant_MOL_temperature
    use integrate_data, only : time_step
   implicit none
   real(8), allocatable :: mol_thermostat_force(:,:),mol_thermostat_torque(:,:)
   integer i,j,k
   real(8) csi_BER_thermo_trans,csi_BER_thermo_rot
   csi_BER_thermo_trans = (Temperature_trans_Calc/temperature - 1.0d0) / thermo_coupling**2 !* time_step
   csi_BER_thermo_rot   = (Temperature_rot_Calc/temperature - 1.0d0) / thermo_coupling**2   !* time_step
   allocate(mol_thermostat_force(Nmols,3),mol_thermostat_torque(Nmols,3))
   do i = 1, Nmols    ;     
   if (.not.l_WALL_MOL_CTRL(i)) then
     mol_thermostat_force(i,:)  =  - csi_BER_thermo_trans * mol_MOM(i,:)
     mol_thermostat_torque(i,:) =  - csi_BER_thermo_rot   * mol_ANG(i,:)
   endif
   enddo         ;       
   do i = 1, Nmols 
   if (.not.l_WALL_MOL_CTRL(i)) then
     mol_force(i,:) = mol_force(i,:) + mol_thermostat_force(i,:)
     mol_torque(i,:) = mol_torque(i,:) + mol_thermostat_torque(i,:)
   endif
   enddo
   deallocate(mol_thermostat_force,mol_thermostat_torque) 
 end subroutine ber_G4_thermo_adjust

 subroutine ber_G4_BARR_XYZ_adjust
   use sim_cel_data
   use physical_constants, only : water_compresibility
   use ALL_mols_data, only : Nmols,l_WALL_MOL_CTRL, mol_force,mol_xyz
   use ensamble_data, only : pressure_xx,pressure_yy,pressure_zz,barostat_coupling
   use stresses_data, only : stress, stress_kin
   use boundaries, only : cel_properties
   use integrate_data
   use kinetics, only : get_mol_stresses
   implicit none
   integer i,j,k
   real(8) t(3), p(3), inv_coupling(3),t_adjust(3), density,local_pressure(10)

       call get_mol_stresses ! kinetic stresses
       local_pressure(1:3) = stress(1:3) + stress_kin(1:3)
       t(1:3) = local_pressure(1:3) / Volume
       inv_coupling(:) = water_compresibility / barostat_coupling(:)
       p(1)  = pressure_xx ; p(2) = pressure_yy ; p(3) = pressure_zz
       t_adjust(:) = (1.0d0+time_step*(t(:)-p(:))*inv_coupling(:))**(1.0d0/3.0d0)
       if (barostat_coupling(1) < 1.0d6) sim_cel(1:3) = t_adjust(1)*sim_cel(1:3) 
       if (barostat_coupling(2) < 1.0d6) sim_cel(4:6) = t_adjust(2)*sim_cel(4:6)  
       if (barostat_coupling(3) < 1.0d6) sim_cel(7:9) = t_adjust(3)*sim_cel(7:9)
       call cel_properties(.true.)
       density=dble(Nmols)/volume
       do i = 1, Nmols
        if (.not.l_WALL_MOL_CTRL(i)) then
           mol_xyz(i,:)= mol_xyz(i,:)*t_adjust(:)
        endif
       enddo
 end subroutine ber_G4_BARR_XYZ_adjust

 end module gear_4_module_MOL
