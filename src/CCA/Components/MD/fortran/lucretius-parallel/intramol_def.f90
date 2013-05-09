
 module intramol_forces_def
 implicit none

 public :: is_bond
 public :: get_style_bond
 public :: set_units_bond
 public :: is_angle
 public :: get_style_angle
 public :: set_units_angle
 public :: is_dihedral
 public :: get_style_dihedral
 public :: set_units_dihedral
 public :: is_deform
 public :: get_style_deform
 public :: set_units_deform 
 contains

! BONDS

  logical function is_bond(ch)
  implicit none
  character(*) ch
  integer N,istyle
    call get_style_bond(ch,N,istyle,.false.)
    is_bond=istyle>0 
  end function is_bond

  subroutine get_style_bond(cstyle,N,istyle,break)
  implicit none
  character(*), intent(IN) :: cstyle
  integer, intent(OUT) :: N, istyle
  logical, optional , intent(IN) :: break
  logical l_break
  
  l_break = .true.
  if (present(break)) l_break=break 
  istyle = -999
  select case (cstyle) 
   case ('HAR')   ! En=k(x-x0)^2 
     N = 2
     istyle = 1
   case default
    if (l_break) then 
     print*, ' ERROR No Style of bond "',cstyle,'" defined in code'
     STOP 
    endif
  end select 
  end subroutine get_style_bond

  subroutine set_units_bond(cstyle,i_type_in_unit, N_prms, the_prms, one_more_dist)  ! set the units to standard input units
  use units_def, only : &
   Def_kJpermol_unitsFlag_CTRL       ,&
   Def_kcalpermol_unitsFlag_CTRL     ,&
   Def_atomicUnits_unitsFlag_CTRL    ,&
   Def_electronVolt_unitsFlag_CTRL   ,&
   Def_Kelvin_unitsFlag_CTRL         ,&
   Def_Internal_unitsFlag_CTRL       ,&
   MX_char_unitsName_len
  use physical_constants, only : calory_to_joule, &
                               eV_to_kJ_per_mol,Kelvin_to_kJ_per_mol,LJ_epsilon_convert ,&
                               Avogadro_number, unit_length
  use Def_atomic_units_module, only : atomic_units
  use math_constants, only : pi

  implicit none
  character(*), intent(IN) :: cstyle
  integer, intent(IN) :: i_type_in_unit
  integer, intent(IN) :: N_prms
  real(8), intent(INOUT) :: the_prms(1:N_prms)
  real(8), intent(INOUT) :: one_more_dist ! the constraint if the bond is constrained.
!  integer, intent(OUT) :: N, istyle
 integer N,istyle
real(8) cvt_en_from_au_to_in, cvt_dist_from_au_to_in

     cvt_en_from_au_to_in = atomic_units%energy * Avogadro_number  / 1000.0d0 ! from a.u to kJ/mol
     cvt_dist_from_au_to_in = atomic_units%length / unit_length ! from a.u to Amstrom


  select case (cstyle)
   case ('HAR')   ! En=k(x-x0)^2
     N = 2
     istyle = 1
     call see_if_OK(istyle,N,N_prms)
      if (i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then! kJ/mol
           the_prms(1) = the_prms(1) * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule * LJ_epsilon_convert! make it in internal units
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in*cvt_dist_from_au_to_in**2 * LJ_epsilon_convert
           the_prms(2) = the_prms(2) * cvt_dist_from_au_to_in
           one_more_dist = one_more_dist * cvt_dist_from_au_to_in
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)   then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1)    / LJ_epsilon_convert * LJ_epsilon_convert
        else
      endif

   case default
     print*, ' ERROR No Style of bond "',cstyle,'" defined in code'
     STOP
  end select

  contains
   subroutine see_if_OK(i,N,N_prms)
   implicit none
   integer, intent(IN) :: i,N,N_prms
   if (N /= N_prms) then
     print*, N,N_prms,'ERROR in see_if_OK%set_bond_units; N is not equal N_prms for bond',i
     STOP
   endif
   end subroutine see_if_OK

  end subroutine set_units_bond


!\ --------------BONDS


! ----------------ANGLES

 logical function is_angle(ch)
  implicit none
  character(*) ch
  integer N,istyle
    call get_style_angle(ch,N,istyle,.false.)
    is_angle=istyle>0
  end function is_angle

  subroutine get_style_angle(cstyle,N,istyle,break)
  implicit none
  character(*), intent(IN) :: cstyle
  integer, intent(OUT) :: N, istyle
  logical, optional , intent(IN) :: break
  logical l_break

  l_break = .true.
  if (present(break)) l_break=break
  istyle = -999

  select case (cstyle)
   case ('HAR')
     N = 2
     istyle = 1
   case('HAR3')  !kx^2/2+kx^3/3+kx^4/4
     N= 4
     istyle = 2  
   case default
   if (l_break) then
     print*, ' ERROR No Style of ANGLE "',cstyle,'" defined in code'
     STOP
   endif
  end select
  end subroutine get_style_angle

  subroutine set_units_angle(cstyle,i_type_in_unit, N_prms, the_prms)  ! set the units to standard input units
  use units_def, only : &
   Def_kJpermol_unitsFlag_CTRL       ,&
   Def_kcalpermol_unitsFlag_CTRL     ,&
   Def_atomicUnits_unitsFlag_CTRL    ,&
   Def_electronVolt_unitsFlag_CTRL   ,&
   Def_Kelvin_unitsFlag_CTRL         ,&
   Def_Internal_unitsFlag_CTRL       ,&
   MX_char_unitsName_len
  use physical_constants, only : calory_to_joule, &
                               eV_to_kJ_per_mol,Kelvin_to_kJ_per_mol,LJ_epsilon_convert ,&
                               Avogadro_number, unit_length
  use Def_atomic_units_module, only : atomic_units
  use math_constants, only : pi
  implicit none
  character(*), intent(IN) :: cstyle
  integer, intent(IN) :: i_type_in_unit
  integer, intent(IN) :: N_prms
  real(8), intent(INOUT) :: the_prms(1:N_prms)
!  integer, intent(OUT) :: N, istyle
 integer N,istyle
real(8) cvt_en_from_au_to_in, cvt_dist_from_au_to_in

     cvt_en_from_au_to_in = atomic_units%energy * Avogadro_number  / 1000.0d0 ! from a.u to kJ/mol
     cvt_dist_from_au_to_in = atomic_units%length / unit_length ! from a.u to Amstrom


 select case (cstyle)
   case ('HAR')
     N = 2
     istyle = 1
     call see_if_OK(istyle,N,N_prms)
      the_prms(2) = the_prms(2) * Pi /180.0d0
      if (i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then! kJ/mol
           the_prms(1) = the_prms(1) * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule * LJ_epsilon_convert ! make it in i.u.
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in*cvt_dist_from_au_to_in**2 * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)   then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1)    / LJ_epsilon_convert * LJ_epsilon_convert
        else
      endif

   case('HAR3')  !kx^2/2+kx^3/3+kx^4/4
     N= 4
     istyle = 2
     call see_if_OK(istyle,N,N_prms)
      the_prms(4) = the_prms(4) * Pi /180.0d0
      if (i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then! kJ/mol
           the_prms(1:3) = the_prms(1:3) * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1:3) = the_prms(1:3) * calory_to_joule * LJ_epsilon_convert ! make it in i.u
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then! a.u.
           the_prms(1:3) = the_prms(1:3) * cvt_en_from_au_to_in*cvt_dist_from_au_to_in**2 * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1:3) = the_prms(1:3) * eV_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)   then
           the_prms(1:3) = the_prms(1:3) * Kelvin_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1:3) = the_prms(1:3)    / LJ_epsilon_convert * LJ_epsilon_convert
        else
      endif

   case default
     print*, ' ERROR No Style of ANGLE "',cstyle,'" defined in code'
     STOP
  end select


  contains
   subroutine see_if_OK(i,N,N_prms)
   implicit none
   integer, intent(IN) :: i,N,N_prms
   if (N /= N_prms) then
     print*, N,N_prms,'ERROR in see_if_OK%set_ANGLE_units; N is not equal N_prms for ANGLE',i
     STOP
   endif
   end subroutine see_if_OK

end subroutine set_units_angle

! \----------------ANGLES


!------------------DIHEDRAL
 logical function is_dihedral(ch)
  implicit none
  character(*) ch
  integer N,istyle
    call get_style_dihedral(ch,N,istyle,.false.)
    is_dihedral=istyle>0
  end function is_dihedral

  subroutine get_style_dihedral(cstyle,N,istyle,break)
  implicit none
  character(*), intent(IN) :: cstyle
  integer, intent(OUT) :: istyle
  integer, intent(INOUT) :: N
  logical, optional , intent(IN) :: break

  logical l_break

  l_break = .true.
  if (present(break)) l_break=break
  istyle = -999
  select case (cstyle)
   case ('COS_N')
!     N = 2
     istyle = 1
   case ('HAR')
     N = 2   !  IMPROPER
     istyle = 2
   case default
   if (l_break) then
     print*, ' ERROR No Style of DIHEDRAL "',cstyle,'" defined in code'
     STOP
   endif
  end select
  end subroutine get_style_dihedral


 subroutine set_units_dihedral(cstyle,i_type_in_unit, N_prms, the_prms)
  use units_def, only : &
   Def_kJpermol_unitsFlag_CTRL       ,&
   Def_kcalpermol_unitsFlag_CTRL     ,&
   Def_atomicUnits_unitsFlag_CTRL    ,&
   Def_electronVolt_unitsFlag_CTRL   ,&
   Def_Kelvin_unitsFlag_CTRL         ,&
   Def_Internal_unitsFlag_CTRL       ,&
   MX_char_unitsName_len
  use physical_constants, only : calory_to_joule, &
                               eV_to_kJ_per_mol,Kelvin_to_kJ_per_mol,LJ_epsilon_convert ,&
                               Avogadro_number, unit_length
  use Def_atomic_units_module, only : atomic_units
  use math_constants, only : pi
   implicit none
   character(*), intent(IN) :: cstyle
   integer, intent(IN) :: i_type_in_unit
   integer, intent(IN) :: N_prms
   real(8), intent(INOUT) :: the_prms(:)
!  integer, intent(OUT) :: N, istyle
   integer N,istyle
   real(8) cvt_en_from_au_to_in, cvt_dist_from_au_to_in

   cvt_en_from_au_to_in = atomic_units%energy * Avogadro_number  / 1000.0d0 ! from a.u to kJ/mol
   cvt_dist_from_au_to_in = atomic_units%length / unit_length ! from a.u to Amstrom

  select case (cstyle)
   case ('HAR')
     N = 2
     istyle = 2
     call see_if_OK(iStyle,N,N_prms)
     the_prms(2) = the_prms(2) * Pi /180.0d0
      if (i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then! kJ/mol
           the_prms(1) = the_prms(1) * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule * LJ_epsilon_convert ! make it in i.u
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)   then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1)    / LJ_epsilon_convert * LJ_epsilon_convert
        else
      endif
   case ('COS_N')   ! CASE 1
!     N = 3
!     N_prms  is the N_fold
! actual number of parameters is Nfodls + 1 more which is the angle 
     istyle = 1
      the_prms(N_prms+1) = the_prms(N_prms+1) * Pi /180.0d0
      if (i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then! kJ/mol
           the_prms(1:N_prms) = the_prms(1:N_prms) * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1:N_prms) = the_prms(1:N_prms) * calory_to_joule * LJ_epsilon_convert ! make it in 
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then! a.u.
           the_prms(1:N_prms) = the_prms(1:N_prms) * cvt_en_from_au_to_in * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1:N_prms) = the_prms(1:N_prms) * eV_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)   then
           the_prms(1:N_prms) = the_prms(1:N_prms) * Kelvin_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1:N_prms) = the_prms(1:N_prms)    / LJ_epsilon_convert * LJ_epsilon_convert
        else
      endif
   case default
     print*, ' ERROR No Style of DIHEDRAL "',cstyle,'" defined in code'
     STOP 
  end select


 contains
   subroutine see_if_OK(i,N,N_prms)
   implicit none
   integer, intent(IN) :: i,N,N_prms
   if (N /= N_prms) then
     print*, N,N_prms,'ERROR in see_if_OK%set_DIHEDRAL_units; N is not equal N_prms for DIEHDRAL',i
     STOP
   endif
   end subroutine see_if_OK

 end subroutine set_units_dihedral
! \----------------DIHEDRAL

! ---- OUT OF PLANE DEFORMATIONS
 logical function is_deform(ch)
  implicit none
  character(*) ch
  integer N,istyle
    call get_style_deform(ch,N,istyle,.false.)
    is_deform=istyle>0
  end function is_deform


  subroutine get_style_deform(cstyle,N,istyle,break)
  implicit none
  character(*), intent(IN) :: cstyle
  integer, intent(OUT) :: istyle
  integer, intent(INOUT) :: N
  logical, optional , intent(IN) :: break

  logical l_break

  l_break = .true.
  if (present(break)) l_break=break
  istyle = -999
  select case (cstyle)
   case ('SMITH')
     N = 2
     istyle = 1
   case default
   if (l_break) then
     print*, ' ERROR No Style of OUT OF PLANE DEFORM "',cstyle,'" defined in code'
     STOP
   endif
  end select
  end subroutine get_style_deform


 subroutine set_units_deform(cstyle,i_type_in_unit, N_prms, the_prms)
  use units_def, only : &
   Def_kJpermol_unitsFlag_CTRL       ,&
   Def_kcalpermol_unitsFlag_CTRL     ,&
   Def_atomicUnits_unitsFlag_CTRL    ,&
   Def_electronVolt_unitsFlag_CTRL   ,&
   Def_Kelvin_unitsFlag_CTRL         ,&
   Def_Internal_unitsFlag_CTRL       ,&
   MX_char_unitsName_len
  use physical_constants, only : calory_to_joule, &
                               eV_to_kJ_per_mol,Kelvin_to_kJ_per_mol,LJ_epsilon_convert ,&
                               Avogadro_number, unit_length
  use Def_atomic_units_module, only : atomic_units
  use math_constants, only : pi
   implicit none
   character(*), intent(IN) :: cstyle
   integer, intent(IN) :: i_type_in_unit
   integer, intent(IN) :: N_prms
   real(8), intent(INOUT) :: the_prms(1:N_prms)
!  integer, intent(OUT) :: N, istyle
   integer N,istyle
   real(8) cvt_en_from_au_to_in, cvt_dist_from_au_to_in

   cvt_en_from_au_to_in = atomic_units%energy * Avogadro_number  / 1000.0d0 ! from a.u to kJ/mol
   cvt_dist_from_au_to_in = atomic_units%length / unit_length ! from a.u to Amstrom

  select case (cstyle)
   case ('SMITH')
     N = 2
     istyle = 1
     call see_if_OK(iStyle,N,N_prms)
     the_prms(2) = the_prms(2) * Pi /180.0d0
      if (i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then! kJ/mol
           the_prms(1) = the_prms(1) * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule * LJ_epsilon_convert ! make it in kJ/mol
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)   then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol * LJ_epsilon_convert
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1)    / LJ_epsilon_convert * LJ_epsilon_convert
        else
      endif
   case default
     print*, ' ERROR No Style of OUT OF PLANE DEFORMS "',cstyle,'" defined in code'
     STOP
  end select

  contains
   subroutine see_if_OK(i,N,N_prms)
   implicit none
   integer, intent(IN) :: i,N,N_prms
   if (N /= N_prms) then
     print*, N,N_prms,'ERROR in see_if_OK%set_outOfPlaneDeforms_units; N is not equal N_prms for DEFORM ',i
     STOP
   endif
   end subroutine see_if_OK


 end subroutine set_units_deform

! \ ---- OUT OF PLANE DEFORMATIONS

 end module intramol_forces_def

