module ensamble_driver_module
 implicit none
 public :: ensamble_driver

 contains
 include 'NVE_VV.f90'
 include 'NVT_B_VV.f90'

 subroutine ensamble_driver(step)
 use ensamble_data
 use integrate_data, only : i_type_integrator_CTRL, l_do_QN_CTRL
 use gear_4_module_MOL
 use sim_cel_data
 use ALL_atoms_data, only : xxx,yyy,zzz,xx,yy,zz,Natoms
 use boundaries, only : periodic_images, adjust_box

    implicit none
    integer, intent(IN) :: step 
if (i_type_integrator_CTRL == 0 ) then   ! VV

        select case (i_type_ensamble_CTRL)
        case(0)
          if (l_do_QN_CTRL) then
print*,'ensamble_driver case 0/0/l_do_QN_CTRL not implemented'; stop
          else
             call NVE_VV(step)
          endif
        case(1) 
          if (l_do_QN_CTRL) then
print*,'ensamble_driver case 0/1/l_do_QN_CTRL not implemented'; stop
          else
             call NVT_B_VV(step)
          endif
!        case(2)
!        case(3)
!        case(4)
        end select ! i_type_ensamble_CTRL
!    xx = xxx; yy = yyy; zz = zzz

else if (i_type_integrator_CTRL == 1 ) then  ! GEAR 4

      select case (i_type_ensamble_CTRL)
        case(0)
          if (l_do_QN_CTRL) then
              call NVE_G4_MOL(step)
          else
             print*,'ensamble_driver case 1/0/.NOT.l_do_QN_CTRL not implemented'; stop
          endif
        case(1)
          if (l_do_QN_CTRL) then
              call NVT_G4_BER_MOL(step)
          else
             print*,'ensamble_driver case 1/1/.NOT.l_do_QN_CTRL not implemented'; stop
          endif
!        case(2)
!        case(3)
!        case(4)
!        case(5)
        case(6)
           if (l_do_QN_CTRL) then
              call NPT_G4_BER_XYZ_MOL(step)
          else
             print*,'ensamble_driver case 1/1/.NOT.l_do_QN_CTRL not implemented'; stop
          endif 
      end select ! i_type_ensamble_CTRL

else

  print*, 'NOT DEFINED ensable in ensamble_driver_module'
  STOP

endif

    call adjust_box
 end subroutine ensamble_driver
end module ensamble_driver_module
