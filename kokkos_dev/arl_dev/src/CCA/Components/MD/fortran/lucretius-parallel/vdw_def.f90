module vdw_def
use mol_type_data
implicit none

public :: is_vdw
public :: get_style_vdw
public :: get_vdw_units
public :: get_vdw_cross_term !discontinued
public :: set_up_vdw_interpol_interact
private :: write_vdw_table

contains

logical function is_vdw(ch)
  character(*) ch
  integer N,style
  call get_style_vdw(ch,N,style,.false.)
  is_vdw = style > 0
end function is_vdw

subroutine get_style_vdw(temp,N,style,break)
character(*), intent(IN) :: temp
integer, intent(OUT) :: N,style
logical, optional, intent(IN) :: break
logical l_break
! n = number of parameters

l_break = .true.
if (present(break)) l_break=break

select  case  (temp)
 case('A12-B6')
   n = 2
   style = 1
!A/r^12-B/r^6
 case ('LJ6-12')
   n=2
   style = 2
 case ('N-M')
! a/(b-c)*(c*(d/r)**b-b*(d/r)**c)
   n = 4
   style = 3
 case ('EXP-6')
! a*exp(-r/b)-c/r**6
   n = 3
   style = 4
 case ('EXP-6-8')
! a*exp(b*(c-r))-d/r**6-e/r**8
   n = 5
   style = 5
 case ('A12-B10')
! zz= a/r**12 - b/r**10
   n = 2
   style = 6
 case ('F-SHIFT-N-M') 
!   a/(b-c)*( c*(b1**b)*((d/r)**b-(1.0d0/c1)**b) - b*(b1**c)*((d/r)**c-(1.0d0/c1)**c)
! + b*c*((r/(c1*d)-1.0d0)*((b1/c1)**b-(b1/c1)**c)) ) 
   n = 6
   style = 7
 case ('HAR-EXP')
! a*((1.0d0-exp(-c*(r-b)))**2-1.0d0) 
   n = 2
   style = 8
 case ('TANG-TOE-6-8-SW') ! TANG TOE 6-8 with swith
   n = 6
   style = 9
 case ('A/R^A+B/R^B') !1-A 2-B 3-a 4-b ; b is negative.
 n = 4
 style = 10
 case ('EXP-6+12')
! a*exp(b*(c-r))-d/r**6+e/r**12
   n = 4
   style = 11
 case ('EXP-6+12+CHEMO') ! Add to exp-6+12 a chemosorbtion term C*exp(r-beta_0)^2/w
  n = 7
  style  = 12
  case default
  if (l_break) then
    print*, 'ERROR in get_style_vdw : Not Defined vdw interaction' , trim(temp)
    STOP
  endif 
end select

end subroutine get_style_vdw

subroutine get_vdw_units(i_type_in_unit,prm_style,N_prms,the_prms)
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
implicit none
integer, intent(IN) :: i_type_in_unit ! 1=kJ/mol 2=kcal/mol 3=a.u 4=ev
integer, intent(IN) :: prm_style
integer, intent(IN) :: N_prms
real(8),intent(INOUT) :: the_prms(1:N_prms)
integer N
real(8) cvt_en_from_au_to_in, cvt_dist_from_au_to_in
real(8) alpha,beta

     cvt_en_from_au_to_in = atomic_units%energy * Avogadro_number  / 1000.0d0 ! from a.u to kJ/mol
     cvt_dist_from_au_to_in = atomic_units%length / unit_length ! from a.u to Amstrom

select case (prm_style) 

 case (1)  !('A12-B6')
   n = 2
   call see_if_OK(1,N,N_prms)
    if (i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1:2) = the_prms(1:2) * calory_to_joule ! make it in kJ/mol
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then! a.u.
           the_prms(1:2) = the_prms(1:2) * cvt_en_from_au_to_in*cvt_dist_from_au_to_in**6
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1:2) = the_prms(1:2) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)   then
           the_prms(1:2) = the_prms(1:2) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1:2) = the_prms(1:2)    / LJ_epsilon_convert
        else
      endif

 case (2)  !('LJ6-12')
   n=2
   call see_if_OK(2,N,N_prms)
    if(i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule ! make it in kJ/mol
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in
           the_prms(2) = the_prms(2) * cvt_dist_from_au_to_in
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)   then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL)  then
           the_prms(1) = the_prms(1) / LJ_epsilon_convert
        else
      endif
 case (3)  !('N-M')
! a/(b-c)*(c*(d/r)**b-b*(d/r)**c)
   n = 4
   call see_if_OK(3,N,N_prms)
        if(i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule ! make it in kJ/mol
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in
           the_prms(4) = the_prms(4) * cvt_dist_from_au_to_in
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1) / LJ_epsilon_convert
        else
        endif

 case (4)  !('EXP-6')
! a*exp(-r/b)-c/r**6
   n= 3
   call see_if_OK(4,N,N_prms)
        if(i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then ! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule ! make it in kJ/mol
           the_prms(3) = the_prms(3) * calory_to_joule
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then ! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in
           the_prms(2) = the_prms(2) * cvt_dist_from_au_to_in
           the_prms(3) = the_prms(3) * cvt_en_from_au_to_in*cvt_dist_from_au_to_in**6
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol
           the_prms(3) = the_prms(3) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)   then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol
           the_prms(3) = the_prms(3) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL)  then
           the_prms(1) = the_prms(1) / LJ_epsilon_convert
           the_prms(3) = the_prms(3) / LJ_epsilon_convert
        else
        endif


 case (5)    !('EXP-6-8')
! a*exp(b*(c-r))-d/r**6-e/r**8
   n = 5
   call see_if_OK(5,N,N_prms)
        if(i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then ! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule ! make it in kJ/mol
           the_prms(4:5) = the_prms(4:5) * calory_to_joule
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then ! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in
           the_prms(2) = the_prms(2) / cvt_dist_from_au_to_in
           the_prms(3) = the_prms(3) * cvt_en_from_au_to_in
           the_prms(4) = the_prms(4) * cvt_en_from_au_to_in*cvt_dist_from_au_to_in**6
           the_prms(5) = the_prms(5) * cvt_en_from_au_to_in*cvt_dist_from_au_to_in**8
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol
           the_prms(4:5) = the_prms(4:5) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL)       then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol
           the_prms(4:5) = the_prms(4:5) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL)      then
            the_prms(1) = the_prms(1) / LJ_epsilon_convert
            the_prms(4:5) = the_prms(4:5) / LJ_epsilon_convert
        else
        endif


 case (6)      !('A12-B10')
! zz= a/r**12 - b/r**10
   n = 2
   call see_if_OK(6,N,N_prms)
        if(i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then ! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1:2) = the_prms(1:2) * calory_to_joule ! make it in kJ/mol
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then ! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in*cvt_dist_from_au_to_in**12
           the_prms(2) = the_prms(2) * cvt_en_from_au_to_in*cvt_dist_from_au_to_in**10
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1:2) = the_prms(1:2) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL) then
           the_prms(1:2) = the_prms(1:2) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1:2) = the_prms(1:2) / LJ_epsilon_convert
        else
        endif

 case (7)      !('F-SHIFT-N-M')
   n = 6
   call see_if_OK(7,N,N_prms)
   print*,'vdw case 7 not implemented; STOP'; STOP 

 case (8)    !('HAR-EXP')
   n = 2
   call see_if_OK(8,N,N_prms)
        if(i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then ! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule ! make it in kJ/mol
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then ! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in
           the_prms(2) = the_prms(2) * cvt_dist_from_au_to_in
           the_prms(3) = the_prms(3) / cvt_dist_from_au_to_in
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL) then 
            the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then 
            the_prms(1) = the_prms(1) / LJ_epsilon_convert
        else
        endif

 case (9) !('TANG-TOE-6-8-SW') ! TANG TOE 6-8 with swith
   n = 6
   call see_if_OK(9,N,N_prms)
        if(i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then ! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule ! make it in kJ/mol
           the_prms(3:4) = the_prms(3:4) * calory_to_joule
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then ! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in
           the_prms(2) = the_prms(2) / cvt_dist_from_au_to_in
           the_prms(3) = the_prms(3) *cvt_en_from_au_to_in* cvt_dist_from_au_to_in**6
           the_prms(4) = the_prms(4) *cvt_en_from_au_to_in* cvt_dist_from_au_to_in**8
           the_prms(5:6) = the_prms(5:6) / cvt_dist_from_au_to_in
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol
           the_prms(3:4) = the_prms(3:4) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol
           the_prms(3:4) = the_prms(3:4) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1) / LJ_epsilon_convert
           the_prms(3:4) = the_prms(3:4) / LJ_epsilon_convert
        else
        endif

 case (10) !('A/R^A+B/R^B') !1-A 2-B 3-a 4-b ; b is negative.
   n = 4
   call see_if_OK(10,N,N_prms)
   alpha = the_prms(1) ; beta = the_prms(2)
        if(i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then ! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(3:4) = the_prms(3:4) * calory_to_joule
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then ! a.u.
           the_prms(3) = the_prms(3)*cvt_en_from_au_to_in* cvt_dist_from_au_to_in**alpha
           the_prms(4) = the_prms(4)*cvt_en_from_au_to_in* cvt_dist_from_au_to_in**beta
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(3:4) = the_prms(3:4) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL) then 
           the_prms(3:4) = the_prms(3:4) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then 
           the_prms(3:4) = the_prms(3:4) / LJ_epsilon_convert
        else
        endif

 case (11) !('EXP-6+12')
   n=4
   call see_if_OK(11,N,N_prms)
        if (i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then ! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule
           the_prms(3:4) = the_prms(3:4) * calory_to_joule
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then ! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in
           the_prms(2) = the_prms(2) / cvt_dist_from_au_to_in
           the_prms(3) = the_prms(3)*cvt_en_from_au_to_in* cvt_dist_from_au_to_in**6
           the_prms(4) = the_prms(4)*cvt_en_from_au_to_in* cvt_dist_from_au_to_in**12
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol
           the_prms(3:4) = the_prms(3:4) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL) then 
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol
           the_prms(3:4) = the_prms(3:4) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then 
           the_prms(1) = the_prms(1) / LJ_epsilon_convert
           the_prms(3:4) = the_prms(3:4) / LJ_epsilon_convert
        else
        endif


 case (12) ! ('EXP+POLY5') exp-6+12 + C*exp(-(r-r0)^2/w)
   n=7
   call see_if_OK(12,N,N_prms)
        if (i_type_in_unit==Def_kJpermol_unitsFlag_CTRL) then ! kJ/mol
        elseif(i_type_in_unit==Def_kcalpermol_unitsFlag_CTRL) then ! kcal/mol
           the_prms(1) = the_prms(1) * calory_to_joule
           the_prms(3:4) = the_prms(3:4) * calory_to_joule
           the_prms(5) = the_prms(5) * calory_to_joule
        elseif(i_type_in_unit==Def_atomicUnits_unitsFlag_CTRL) then ! a.u.
           the_prms(1) = the_prms(1) * cvt_en_from_au_to_in
           the_prms(2) = the_prms(2) / cvt_dist_from_au_to_in
           the_prms(3) = the_prms(3)*cvt_en_from_au_to_in* cvt_dist_from_au_to_in**6
           the_prms(4) = the_prms(4)*cvt_en_from_au_to_in* cvt_dist_from_au_to_in**12
           the_prms(5) = the_prms(5) * cvt_en_from_au_to_in
           the_prms(6:7) = the_prms(6:7) / cvt_dist_from_au_to_in
        elseif(i_type_in_unit==Def_electronVolt_unitsFlag_CTRL) then ! eV
           the_prms(1) = the_prms(1) * eV_to_kJ_per_mol
           the_prms(3:4) = the_prms(3:4) * eV_to_kJ_per_mol
           the_prms(5) = the_prms(5) * eV_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Kelvin_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1) * Kelvin_to_kJ_per_mol
           the_prms(3:4) = the_prms(3:4) * Kelvin_to_kJ_per_mol
           the_prms(5) = the_prms(5) * Kelvin_to_kJ_per_mol
        elseif(i_type_in_unit==Def_Internal_unitsFlag_CTRL) then
           the_prms(1) = the_prms(1) / LJ_epsilon_convert
           the_prms(3:4) = the_prms(3:4) / LJ_epsilon_convert
           the_prms(5) = the_prms(5) / LJ_epsilon_convert
           the_prms(6:7) = the_prms(6:7) / LJ_epsilon_convert 
        else
        endif

  case default
  print*, 'ERROR in get_vdw_units : Not Defined vdw interaction' ,prm_style
  STOP
end select


contains
subroutine see_if_OK(i,N,N_prms)
implicit none
integer, intent(IN) :: i,N,N_prms
 if (N /= N_prms) then
   print*, N,N_prms,'ERROR in see_if_OK%get_vdw_units; N is not equal N_prms for vdw',i
   STOP
 endif
end subroutine see_if_OK
end subroutine get_vdw_units


subroutine get_vdw_cross_term(style,MX,n1,p1,p2,p3)
integer, intent(IN) :: n1,MX,style
real(8), intent(IN) :: p1(MX),p2(MX)
real(8), intent(OUT) :: p3(MX)
real(8) s1,s2,e1,e2,s12,e12
logical l
select case (style)

case (1)  ! A6B12 p1/r^12 - p2/r^6

 if ((dabs(p1(1)) < 1.0d-9.and.dabs(p1(2)) > 1.0d-9).or.(dabs(p1(1)) > 1.0d-9.and.dabs(p1(2)) < 1.0d-9 )) then
   print*, 'Incorect vdw parameters (one is zero and one is not zero;',p1(1),p1(2),&
           ' and a cross interaction cannot be evaluated;  define their cross interaction manually'
   print*, 'program will stop in get_vdw_cross_term style case 1'
   STOP
 endif 
 if ((dabs(p2(1)) < 1.0d-9.and.dabs(p2(2)) > 1.0d-9).or.(dabs(p2(1)) > 1.0d-9.and.dabs(p2(2)) < 1.0d-9)) then
   print*, 'Incorect vdw parameters (one is zero and one is not zero;',p2(1),p2(2),&
           ' and a cross interaction cannot be evaluated; define their cross interaction manually'
   print*, 'program will stop in get_vdw_cross_term style case 1'
   STOP
 endif

 l = (dabs(p1(1)) < 1.0d-9.and.dabs(p1(2)) < 1.0d-9 ).or.(dabs(p2(1)) < 1.0d-9.and.dabs(p2(2)) < 1.0d-9 )
   if ( l ) then 
       p3(1) =0.0d0
       p3(2) = 0.0d0
    RETURN
   endif
 s1 = (p1(1)/p1(2))**(1.0d0/6.0d0)
 s2 = (p2(1)/p2(2))**(1.0d0/6.0d0)
 e1 = p1(2)/4.0d0/s1**6.0d0
 e2 = p2(2)/4.0d0/s2**6.0d0
 e12 = dsqrt(e1*e2)
 s12 = 0.5d0*(s1+s2)
 p3(1) = 4.0d0*e12*s12**12.0d0
 p3(2) = 4.0d0*e12*s12**6.0d0

case(2)
! first is sigma second is epsilon
 p3(2) = 0.5d0*(p2(2)+p1(2))
 p3(1) = dsqrt(p2(1)*p1(1))
case default
end select
end subroutine get_vdw_cross_term

 subroutine set_up_vdw_interpol_interact
! long range corrections are here 
  use interpolate_data
  use atom_type_data, only : atom_Style2_vdwPrm,atom_Style2_vdwStyle, N_STYLE_ATOMS, pair_which_style
  use physical_constants
  use cut_off_data
  use Def_atomic_units_module, only : atomic_units
  use CTRLs_data, only : i_type_unit_vdw_input_CTRL
use LR_corrections_data  ! long range corrections
use math_constants, only : Pi2
  implicit none
  real(8) rrr
  real(8) dlr_pot,r0, cut
  real(8) gamma, am, an, a0, beta, eps, alpha, v_chem, g_chem,dz
  integer i,j,k,N,ii,N2,is,js
  real(8), allocatable :: prm_vdw(:,:)
  integer MAX_grid_short_range
  real(8) cvt_en_from_au_to_in, cvt_dist_from_au_to_in

     cvt_en_from_au_to_in = atomic_units%energy * Avogadro_number  / 1000.0d0 ! from a.u to kJ/mol
     cvt_dist_from_au_to_in = atomic_units%length / unit_length ! from a.u to Amstrom

  allocate(prm_vdw( lbound(atom_Style2_vdwPrm,dim=1):ubound(atom_Style2_vdwPrm,dim=1) , &
                    lbound(atom_Style2_vdwPrm,dim=2):ubound(atom_Style2_vdwPrm,dim=2) ) ) 
  prm_vdw=0.0d0
! morse potential
  MAX_grid_short_range = MX_interpol_points
  RDR = (cut_off+displacement)/dble(MAX_grid_short_range-4) 
  iRDR = 1.0d0/RDR
  dlr_pot = RDR
N2 = ((N_STYLE_ATOMS+1)*N_STYLE_ATOMS)/2


 atom_Style2_vdwPrm(0,:) = 1.0d6
 
 cut = cut_off

  do ii = 1 , N2
      is = pair_which_style(ii)%i ; js = pair_which_style(ii)%j
      select case (atom_Style2_vdwStyle(ii) ) 
       case(0)
       case(1)
! A/r^12 - B/r^6
            prm_vdw(1,ii) = atom_Style2_vdwPrm(1,ii)*LJ_epsilon_convert
            prm_vdw(2,ii) = atom_Style2_vdwPrm(2,ii)* LJ_epsilon_convert
            if (atom_Style2_vdwPrm(1,ii) ==0.0d0.and.atom_Style2_vdwPrm(2,ii)==0.0d0) atom_Style2_vdwPrm(0,ii)=0.0d0
         do i =  1, MAX_grid_short_range
            rrr = dble(i)*dlr_pot
            vvdw(i,ii)=vv1(rrr,prm_vdw(1,ii),prm_vdw(2,ii) )
            gvdw(i,ii)=gg1(rrr,prm_vdw(1,ii),prm_vdw(2,ii) )
         enddo
         EN0_LR_vdw(is,js) = U_lr_1(cut,prm_vdw(1,ii),prm_vdw(2,ii))
         STRESS0_LR_vdw(is,js) = P_lr_1(cut,prm_vdw(1,ii),prm_vdw(2,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)
       case(2)
!4*eps*( (sigma/r)^12-(sigma/r)^6 ) 
            prm_vdw(1,ii) = atom_Style2_vdwPrm(1,ii) * LJ_epsilon_convert
            prm_vdw(2,ii) = atom_Style2_vdwPrm(2,ii)  ! sigma
            if (atom_Style2_vdwPrm(1,ii) ==0.0d0) atom_Style2_vdwPrm(0,ii) = 0.0d0
         do i =  1, MAX_grid_short_range
            rrr = dble(i)*dlr_pot
            vvdw(i,ii)=vv2(rrr,prm_vdw(1,ii),prm_vdw(2,ii) )
            gvdw(i,ii)=gg2(rrr,prm_vdw(1,ii),prm_vdw(2,ii) )
         enddo
         EN0_LR_vdw(is,js) = U_lr_2(cut,prm_vdw(1,ii),prm_vdw(2,ii))
         STRESS0_LR_vdw(is,js) = P_lr_2(cut,prm_vdw(1,ii),prm_vdw(2,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)
       case(3)
!=a/(b-c)*(c*(d/r)**b-b*(d/r)**c)
         prm_vdw(1,ii) = atom_Style2_vdwPrm(1,ii) * LJ_epsilon_convert
         if (atom_Style2_vdwPrm(1,ii)==0.0d0) atom_Style2_vdwPrm(0,ii)=0.0d0
         do i =  1, MAX_grid_short_range
            rrr = dble(i)*dlr_pot
            vvdw(i,ii)=vv3(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii) )
            gvdw(i,ii)=gg3(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii) )
         enddo
         EN0_LR_vdw(is,js) = U_lr_3(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         STRESS0_LR_vdw(is,js) = P_lr_3(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)

       case(4)
! bukiman exp-6 :    zz=a*exp(-r/b)-c/r**6
         prm_vdw(1,ii)=atom_Style2_vdwPrm(1,ii)*LJ_epsilon_convert
         prm_vdw(3,ii)=atom_Style2_vdwPrm(3,ii)*LJ_epsilon_convert
         if (atom_Style2_vdwPrm(1,ii)==0.0d0.and.atom_Style2_vdwPrm(3,ii)==0.0d0) atom_Style2_vdwPrm(0,ii)=0.0d0
         do i =  1, MAX_grid_short_range
            rrr = dble(i)*dlr_pot
            vvdw(i,ii)=vv4(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii))
            gvdw(i,ii)=gg4(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii))
         enddo
         EN0_LR_vdw(is,js) = U_lr_4(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii))
         STRESS0_LR_vdw(is,js) = P_lr_4(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)

       case(5)
!        born-huggins-meyer exp-6-8   a*exp(b*(c-r))-d/r**6-e/r**8
         prm_vdw(1,ii) =  atom_Style2_vdwPrm(1,ii) * LJ_epsilon_convert
         prm_vdw(4:5,ii)=atom_Style2_vdwPrm(4:5,ii)* LJ_epsilon_convert 
         if (atom_Style2_vdwPrm(1,ii)==0.0d0.and.atom_Style2_vdwPrm(4,ii)==0.0d0.and.atom_Style2_vdwPrm(5,ii)==0.0d0) &
                                                                                            atom_Style2_vdwPrm(0,ii)=0.0d0 
         do i =  1, MAX_grid_short_range
            rrr = dble(i)*dlr_pot
            vvdw(i,ii)=vv5(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),&
                           prm_vdw(4,ii),prm_vdw(5,ii) )
            gvdw(i,ii)=gg5(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),&
                           prm_vdw(4,ii),prm_vdw(5,ii) )
         enddo
         EN0_LR_vdw(is,js) = U_lr_5(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii),prm_vdw(5,ii))
         STRESS0_LR_vdw(is,js) = P_lr_5(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii),prm_vdw(5,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)

       case(6)
! zz= a/r**12 - b/r**10
           prm_vdw(1:2,ii) =  atom_Style2_vdwPrm(1:2,ii) * LJ_epsilon_convert
           if (atom_Style2_vdwPrm(1,ii)==0.0d0.and.atom_Style2_vdwPrm(2,ii)==0.0d0)atom_Style2_vdwPrm(0,ii)=0.0d0
         do i =  1, MAX_grid_short_range
            rrr = dble(i)*dlr_pot
            vvdw(i,ii)=vv6(rrr,prm_vdw(1,ii),prm_vdw(2,ii) )
            gvdw(i,ii)=gg6(rrr,prm_vdw(1,ii),prm_vdw(2,ii) )
         enddo
         EN0_LR_vdw(is,js) = U_lr_6(cut,prm_vdw(1,ii),prm_vdw(2,ii))
         STRESS0_LR_vdw(is,js) = P_lr_6(cut,prm_vdw(1,ii),prm_vdw(2,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)

       case(7)
         prm_vdw(1,ii) = atom_Style2_vdwPrm(1,ii) * LJ_epsilon_convert
         a0=prm_vdw(4,ii)
         an=prm_vdw(2,ii)
         am=prm_vdw(3,ii)
         eps=prm_vdw(1,ii)
         if (an <= am) then
           print*, 'ERROR : In set_up_all_types_of_vdw CASE 7 the an is SMALLER than am and it '
           print*, 'should be otherwise : STOP'
           STOP
         endif
         gamma = cut_off/r0
         if (gamma < 1.0d0 ) then
             print*, 'ERROR : In set_up_all_types_of_vdw CASE 7 the gamma is smaller than 1 and it '
             print*, 'should be otherwise : STOP'
           STOP
         endif
        beta = gamma*((gamma**(am+1.0d0)-1.0d0) /                       &
                      (gamma**(an+1.0d0)-1.0d0))**(1.0d0/(an-am))
        alpha= -(an-am) /                                                 &
                ( am*(beta**an)*(1.0d0+(an/gamma-an-1.0d0)/gamma**an)   &
                - an*(beta**am)*(1.0d0+(am/gamma-am-1.0d0)/gamma**am) )
        eps = eps*alpha
         do i =  1, MAX_grid_short_range
            rrr = dble(i)*dlr_pot
            vvdw(i,ii)=vv7(rrr,eps,an,am,r0,beta,gamma)
            gvdw(i,ii)=gg7(rrr,eps,an,am,r0,beta,gamma)
         enddo
         EN0_LR_vdw(is,js) = U_lr_7(cut,eps,an,am,r0,beta,gamma)
         STRESS0_LR_vdw(is,js) = P_lr_7(cut,eps,an,am,r0,beta,gamma)
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)

       case(8)
! zz=a*((1.0d0-exp(-c*(r-b)))**2-1.0d0)
          prm_vdw(1,ii) = atom_Style2_vdwPrm(1,ii) * LJ_epsilon_convert
          if(atom_Style2_vdwPrm(1,ii)==0.0d0)atom_Style2_vdwPrm(0,ii)=0.0d0
         do i =  1, MAX_grid_short_range
            rrr = dble(i)*dlr_pot
            vvdw(i,ii)=vv8(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii))
            gvdw(i,ii)=gg8(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii))
         enddo
         EN0_LR_vdw(is,js) = U_lr_8(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii))
         STRESS0_LR_vdw(is,js) = P_lr_8(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)

        case(9) 
!  zz=A*dexp(-(r*B)) + C6/r**6*f6 + C8/r**8*f8     f6 = f(b6*r) f8 = f(b8*r)
           prm_vdw(1,ii)=atom_Style2_vdwPrm(1,ii)*LJ_epsilon_convert
 !   prm_vdw(2,) is in Amstrom at -1
 !   prm_vdw(5,) and prm_vdw(6,) are b6 and b8 in 1/Amtrom
           prm_vdw(2,ii)=atom_Style2_vdwPrm(2,ii)
           prm_vdw(3,ii)=atom_Style2_vdwPrm(3,ii)*LJ_epsilon_convert ! C6
           prm_vdw(4,ii)=atom_Style2_vdwPrm(4,ii)*LJ_epsilon_convert ! C8
           prm_vdw(5,ii) = atom_Style2_vdwPrm(5,ii)
           prm_vdw(6,ii) = atom_Style2_vdwPrm(6,ii)
         if (atom_Style2_vdwPrm(1,ii)==0.0d0.and.atom_Style2_vdwPrm(3,ii)==0.0d0.and.atom_Style2_vdwPrm(4,ii)==0.0d0) &
                          atom_Style2_vdwPrm(0,ii)=0.0d0
         do i = 1, MAX_grid_short_range
           rrr = dble(i)*dlr_pot
           vvdw(i,ii)=vv9(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii),prm_vdw(5,ii),prm_vdw(6,ii))
           gvdw(i,ii)=gg9(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii),prm_vdw(5,ii),prm_vdw(6,ii))
         enddo   
         EN0_LR_vdw(is,js) = U_lr_9(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii),prm_vdw(5,ii),prm_vdw(6,ii))
         STRESS0_LR_vdw(is,js) = P_lr_9(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii),prm_vdw(5,ii),prm_vdw(6,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)

     case(10) ! A/r^alpha + B/r*beta ; 
        if (atom_Style2_vdwPrm(3,ii)==0.0d0.and.atom_Style2_vdwPrm(4,ii)==0.0d0)atom_Style2_vdwPrm(0,ii)=0.0d0
        prm_vdw(1:2,ii) = atom_Style2_vdwPrm(1:2,ii)
        prm_vdw(3:4,ii) = atom_Style2_vdwPrm(3:4,ii) * LJ_epsilon_convert ! Get the enegry in internal units
        
        do i =  1, MAX_grid_short_range
         rrr = dble(i)*dlr_pot
         vvdw(i,ii)=vv10(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         gvdw(i,ii)=gg10(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
        enddo
         EN0_LR_vdw(is,js) = U_lr_10(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         STRESS0_LR_vdw(is,js) = P_lr_10(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)

    case(11) ! exp-6+12
           prm_vdw(1,ii)=atom_Style2_vdwPrm(1,ii)*LJ_epsilon_convert
           prm_vdw(2,ii)=atom_Style2_vdwPrm(2,ii)
           prm_vdw(3,ii)=atom_Style2_vdwPrm(3,ii)*LJ_epsilon_convert ! C6
           prm_vdw(4,ii)=atom_Style2_vdwPrm(4,ii)*LJ_epsilon_convert ! C12
         if (atom_Style2_vdwPrm(1,ii)==0.0d0.and.atom_Style2_vdwPrm(3,ii)==0.0d0.and.atom_Style2_vdwPrm(4,ii)==0.0d0) &
                          atom_Style2_vdwPrm(0,ii)=0.0d0
         do i = 1, MAX_grid_short_range
           rrr = dble(i)*dlr_pot
           vvdw(i,ii)=vv11(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
           gvdw(i,ii)=gg11(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         enddo

         EN0_LR_vdw(is,js) = U_lr_11(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         STRESS0_LR_vdw(is,js) = P_lr_11(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)

    case(12) ! exp-6+12 + C*exp(-(r-beta0)^2/w)
           prm_vdw(1,ii)=atom_Style2_vdwPrm(1,ii)*LJ_epsilon_convert
           prm_vdw(2,ii)=atom_Style2_vdwPrm(2,ii)
           prm_vdw(3,ii)=atom_Style2_vdwPrm(3,ii)*LJ_epsilon_convert ! C6
           prm_vdw(4,ii)=atom_Style2_vdwPrm(4,ii)*LJ_epsilon_convert ! C12
           prm_vdw(5,ii)=atom_Style2_vdwPrm(5,ii)*LJ_epsilon_convert
           prm_vdw(6:7,ii)=atom_Style2_vdwPrm(6:7,ii)
         if (atom_Style2_vdwPrm(1,ii)==0.0d0.and.atom_Style2_vdwPrm(3,ii)==0.0d0.and.atom_Style2_vdwPrm(4,ii)==0.0d0.and.atom_Style2_vdwPrm(5,ii)==0.0d0) &
                          atom_Style2_vdwPrm(0,ii)=0.0d0
         do i = 1, MAX_grid_short_range
           rrr = dble(i)*dlr_pot
           dz = rrr-prm_vdw(6,ii)
           v_chem = prm_vdw(5,ii)*dexp(-dz*dz/prm_vdw(7,ii))
           g_chem = v_chem * 2.0d0 * dz / prm_vdw(7,ii) * rrr
           vvdw(i,ii)=vv11(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii)) + v_chem
           gvdw(i,ii)=gg11(rrr,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii)) + g_chem
         enddo

         EN0_LR_vdw(is,js) = U_lr_11(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         STRESS0_LR_vdw(is,js) = P_lr_11(cut,prm_vdw(1,ii),prm_vdw(2,ii),prm_vdw(3,ii),prm_vdw(4,ii))
         EN0_LR_vdw(js,is) = EN0_LR_vdw(is,js); STRESS0_LR_vdw(js,is) = STRESS0_LR_vdw(is,js)


       case default
        print*, ii,'VDW not defned in set_up_all_Styles_of_vdwi in SYSDEF',atom_Style2_vdwStyle(ii)
        STOP
    end select
  enddo !  ii

 
 deallocate (prm_vdw)

 vvdw(0,:)=vvdw(1,:)
 gvdw(0,:)=gvdw(1,:) ! that is NOT going to matter anyway

 call write_vdw_table

 contains

   ! 12-6 potential

  function vv1(r,a,b) result (ZZ) 
   real(8) r,a,b,zz
   ZZ=(a/r**6-b)/r**6
  end function vv1
  function gg1(r,a,b) result (ZZ)
   real(8) r,a,b,zz
   ZZ=6.0d0*(2.0d0*a/r**6-b)/r**6
  end function gg1
  function U_lr_1(r,a,b) result (ZZ)  ! long range correction to potential 
  real(8) r,a,b,zz
   zz = a/(9.0d0*r**9) - b/(3.0d0*r**3)
   zz = zz * Pi2
  end function U_lr_1
  function P_lr_1(r,a,b) result (ZZ)  ! long range correction to pressure
  real(8) r,a,b,zz
   zz =  12.0d0*a/(9.0d0*r**9) - 6.0d0*b/(3.0d0*r**3)
   zz = zz * Pi2
  end function P_lr_1


! lennard-jones potential

  
  function vv2(r,a,b) result (ZZ) 
  real(8) r,a,b,zz
  ZZ=4.0d0*a*(b/r)**6*((b/r)**6-1.0d0)
!print*, r,a,b,(b/r)**6,4.0d0*a*(b/r)**6*((b/r)**6-1.0d0),ZZ
!STOP
  end function vv2
  function gg2(r,a,b) result (ZZ) 
  real(8) r,a,b,zz
  zz=24.0d0*a*(b/r)**6*(2.0d0*(b/r)**6-1.0d0)
  end function gg2
    function U_lr_2(r,a,b) result (ZZ)  ! long range correction to potential
  real(8) r,a,b,zz
   zz = 4.0d0*a*(b**12/(9.0d0*r**9) - b**6/(3.0d0*r**3))
   zz = zz * Pi2
  end function U_lr_2
  function P_lr_2(r,a,b) result (ZZ)  ! long range correction to pressure
  real(8) r,a,b,zz
   zz = 4.0d0*a*(12.0d0*b**12/(9.0d0*r**9) - 2.0d0*b**6/(r**3))
   zz = zz * Pi2
  end function P_lr_2

! n-m potential

  function vv3(r,a,b,c,d) result (ZZ) 
   real(8) r,a,b,c,d,ZZ
    ZZ=a/(b-c)*(c*(d/r)**b-b*(d/r)**c)
  end function vv3
  function gg3(r,a,b,c,d) result (zz) 
   real(8) r,a,b,c,d,ZZ 
   zz=a*c*b/(b-c)*((d/r)**b-(d/r)**c)
  end function gg3
  function U_lr_3(r,a,b,c,d) result (ZZ)  ! long range correction to potential
  real(8) r,a,b,c,d,zz
   zz = a/(b-c)*( c*d**b/((b-3.0d0)*r**(b-3.0d0)) - b*d**c/((c-3.0d0)*r**(c-3.0d0)) )
   zz = zz * Pi2
  end function U_lr_3
  function P_lr_3(r,a,b,c,d) result (ZZ)  ! long range correction to pressure
  real(8) r,a,b,c,d,zz
   zz =  a/(b-c)*b*c*( d**b/((b-3.0d0)*r**(b-3.0d0))  - d**c/((c-3.0d0)*r**(c-3.0d0)) )
   zz = zz * Pi2
  end function P_lr_3

! buckingham exp-6 potential

  function vv4(r,a,b,c)  result (zz)
    real(8) r,a,b,c, zz
    zz=a*exp(-r/b)-c/r**6
  end function vv4
  function  gg4(r,a,b,c) result (zz) 
   real(8) r,a,b,c ,zz
   zz=r*a*exp(-r/b)/b-6.0d0*c/r**6
  end function gg4
   function U_lr_4(r,a,b,c) result (ZZ)  ! long range correction to potential
  real(8) r,a,b,c ,zz,y
   y = r/b
   zz = -c/(R**3*3.0d0)
   zz = zz * Pi2
  end function U_lr_4
  function P_lr_4(r,a,b,c) result (ZZ)  ! long range correction to pressure
  real(8) r,a,b,c,zz,y
   y = r/b
   zz = - 2.0d0*c/R**3
   zz = zz * Pi2
  end function P_lr_4

! born-huggins-meyer exp-6-8 potential

  function vv5(r,a,b,c,d,e)  result (zz)
  real(8) r,a,b,c,d,e,zz
   zz =a*exp(b*(c-r))-d/r**6-e/r**8
  end function vv5
  function gg5(r,a,b,c,d,e) result (zz)
  real(8) r,a,b,c,d,e,zz
  zz=r*a*b*exp(b*(c-r))-6.0d0*d/r**6-8.0d0*e/r**8
  end function gg5
    function U_lr_5(r,a,b,c,d,e) result (ZZ)  ! long range correction to potential
  real(8) r,a,b,c,d,e,zz
   zz = -d/(3.0d0*r**3) - e/(5.0d0*r**5)
   zz = zz * Pi2
  end function U_lr_5
  function P_lr_5(r,a,b,c,d,e) result (ZZ)  ! long range correction to pressure
  real(8) r,a,b,c,d,e,zz
   zz = -2.0d0*d/(r**3) - 8.0d0*e/(5.0d0*r**5)
   zz = zz * Pi2
  end function P_lr_5

! Hydrogen-bond 12-10 potential

  function vv6(r,a,b)  result (zz) 
  real(8) r,a,b ,zz
  zz= a/r**12 - b/r**10
  end function vv6
  function gg6(r,a,b) result(zz) 
   real(8) r,a,b ,zz
   zz= 12.0d0*a/r**12 - 10.0d0*b/r**10
  end function gg6
   function U_lr_6(r,a,b) result (ZZ)  ! long range correction to potential
  real(8) r,a,b,zz
   zz = a/(9.0d0*r**9) - b/(7.0d0*r**7)
   zz = zz * Pi2
  end function U_lr_6
  function P_lr_6(r,a,b) result (ZZ)  ! long range correction to pressure
  real(8) r,a,b,zz
   zz = 12.0d0*a/(9.0d0*r**9) + 1.d1*b/(7.0d0*r**7)
   zz = zz * Pi2
  end function P_lr_6


! shifted and force corrected n-m potential (w. smith)

  function vv7(r,a,b,c,d,b1,c1) result(zz) 
   real(8) r,a,b,c,d,b1,c1,zz 
   zz=a/(b-c)*( c*(b1**b)*((d/r)**b-(1.0d0/c1)**b)      &
      -b*(b1**c)*((d/r)**c-(1.0d0/c1)**c)      &
      +b*c*((r/(c1*d)-1.0d0)*((b1/c1)**b-(b1/c1)**c)) )
  end function vv7
  function gg7(r,a,b,c,d,b1,c1) result(zz) 
   real(8) r,a,b,c,d,b1,c1,zz 
    zz=a*c*b/(b-c)*( (b1**b)*(d/r)**b-(b1**c)*(d/r)**c    &
                       -r/(c1*d)*((b1/c1)**b-(b1/c1)**c) )
  end function gg7
! NO LR correction added here
  function U_lr_7(r,a,b,c,d,b1,c1) result (ZZ)  ! long range correction to potential
  real(8) r,a,b,c,d,b1,c1,zz
   zz = 0.0d0
  end function U_lr_7
  function P_lr_7(r,a,b,c,d,b1,c1) result (ZZ)  ! long range correction to pressure
  real(8) r,a,b,c,d,b1,c1,zz
   zz = 0.0d0
  end function P_lr_7

 
  function vv8(r,a,b,c) result(zz) 
  real(8) r,a,b,c,zz 
  zz=a*((1.0d0-exp(-c*(r-b)))**2-1.0d0)
  end function vv8
  function gg8(r,a,b,c) result(zz) 
  real(8) r,a,b,c,zz 
  zz=-2.0d0*r*a*c*(1.0d0-exp(-c*(r-b)))*exp(-c*(r-b))
  end function gg8
! NO LR correction added here
  function U_lr_8(r,a,b,c) result (ZZ)  ! long range correction to potential
  real(8) r,a,b,c,zz
   zz = 0.0d0
  end function U_lr_8
  function P_lr_8(r,a,b,c) result (ZZ)  ! long range correction to pressure
  real(8) r,a,b,c,zz
   zz = 0.0d0
  end function P_lr_8

! Tang Toenis +6+8 (exp+6f+8f)
  function vv9(r,A,B,C6,C8,b6,b8) result(zz)
  real(8) r,A,B,C6,C8,b6,b8,zz,x,f6,f8
  x =  b6*r
  f6 = 1.0d0 + x + x**2/2.0d0 + x**3 / 6.0d0 + x**4/ 24.0d0 + x**5/120.0d0 + x**6/720.0d0
  f6 = 1.0d0 - dexp(-x) * f6
  x = b8*r
  f8 = 1.0d0 + x + x**2/2.0d0 + x**3 / 6.0d0 + x**4/ 24.0d0 + x**5/120.0d0 + x**6/720.0d0 + &
       x**7/5040.0d0 + x**8/40320.0d0
  f8 = 1.0d0 - dexp(-x) * f8
  zz=A*dexp(-(r*B)) + C6/r**6*f6 + C8/r**8*f8
!print*, '\\\-----',A*dexp(-(r/B)), C6/r**6*f6,C8/r**8*f8,C6/r**6*f6+C8/r**8*f8,zz,'sw:',f6,f8,'\\\\\\======'
  end function vv9
  function gg9(r,A,B,C6,C8,b6,b8) result(zz)
  real(8) r,A,B,C6,C8,b6,b8,zz,x,f6,f8,f6d,f8d, f6_0,f8_0,f6d_times_r,f8d_times_r
  real(8) exp_fct
   x =  b6*r
   exp_fct =  dexp(-x)
   f6_0 = 1.0d0 + x + x**2/2.0d0 + x**3 / 6.0d0 + x**4/ 24.0d0 + x**5/120.0d0 + x**6/720.0d0
   f6 = 1.0d0 - exp_fct * f6_0
   f6d_times_r = x**7/720.0d0 * exp_fct
   x = b8*r
   exp_fct =  dexp(-x)
   f8_0 = 1.0d0 + x + x**2/2.0d0 + x**3 / 6.0d0 + x**4/ 24.0d0 + x**5/120.0d0 + x**6/720.0d0 + &
       x**7/5040.0d0 + x**8/40320.0d0
   f8 = 1.0d0 - exp_fct * f8_0
   f8d_times_r = x**9/40320.0d0 * exp_fct
   zz=A*B*r*dexp(-(r*B)) + 6.0d0*C6/r**6*f6 + 8.0d0*C8/r**8*f8 - (C6/r**6*f6d_times_r + C8/r**8*f8d_times_r)
  end function gg9
  function U_lr_9(r,A,B,C6,C8,b6,b8) result (ZZ)  ! long range correction to potential
  real(8) r,A,B,C6,C8,b6,b8,zz
   zz = C6/(R**3*3.0d0)+C8/(R**5*5.0d0)
   zz = zz * Pi2
  end function U_lr_9
  function P_lr_9(r,A,B,C6,C8,b6,b8) result (ZZ)  ! long range correction to pressure
  real(8) r,A,B,C6,C8,b6,b8,zz
   zz = 2.0d0*C6/R**3+8.0d0/5.0d0*C8/R**5
   zz = zz * Pi2
  end function P_lr_9



  function vv10(r,A,B,C_A,C_B) result(zz)
  real(8) r,A,B,C_A,C_B,zz
  zz = C_A/r**A + C_B/r**B
  end function vv10
  function gg10(r,A,B,C_A,C_B) result(zz)
  real(8) r,A,B,C_A,C_B,zz
  zz =  (A*C_A/r**A + B*C_B/r**B)
  end function gg10
! this is for 3-body terms and stuff; NO LONG RANGE correction will be added
   function U_lr_10(r,A,B,C_A,C_B) result (ZZ)  ! long range correction to potential
  real(8) r,A,B,C_A,C_B,zz
   zz = 0.0d0
  end function U_lr_10
  function P_lr_10(r,A,B,C_A,C_B) result (ZZ)  ! long range correction to pressure
  real(8) r,A,B,C_A,C_B,zz
   zz = 0.0d0
  end function P_lr_10


! exp - 6 + 12
  function vv11(r,A,B,C6,C12) result(zz)
  real(8) r,A,B,C6,C12,zz,r2
    r2 = r*r
    zz = A*exp(-B*r) - C6/r**6 + C12/r**12
  end function vv11
  function gg11(r,A,B,C6,C12) result(zz)
  real(8) r,A,B,C6,C12,zz
   zz=A*B*r*dexp(-(r*B)) - 6.0d0*C6/r**6 + 12.0d0*C12/r**12
  end function gg11
  function U_lr_11(r,A,B,C6,C12) result (ZZ)  ! long range correction to potential
  real(8) r,A,B,C6,C12,zz
   zz = -C6/(R**3*3.0d0)+C12/(R**9*9.0d0)
   zz = zz * Pi2
  end function U_lr_11
  function P_lr_11(r,A,B,C6,C12) result (ZZ)  ! long range correction to pressure
  real(8) r,A,B,C6,C12,zz
   zz = -2.0d0*C6/R**3+4.0d0*C12/(R**9*3.0d0)
   zz = zz * Pi2
  end function P_lr_11



 end subroutine set_up_vdw_interpol_interact


  subroutine write_vdw_table
  use interpolate_data
  use cut_off_data
! dlr_pot must be known
  implicit none
  integer i

  open(unit=33,file='./runs/vdw_table.txt',recl=32*(ubound(vvdw,dim=2)*2+10))  
  do i =  1, MX_interpol_points
          write(33,*) dble(i)*rdr, vvdw(i,:) * 0.01d0  ! in kJ/mol
  enddo
  close(33)  
  end subroutine write_vdw_table

end module vdw_def
