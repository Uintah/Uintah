
module units_def
implicit none
integer ::  Def_kJpermol_unitsFlag_CTRL    = 1
integer ::  Def_kcalpermol_unitsFlag_CTRL  = 2
integer ::  Def_atomicUnits_unitsFlag_CTRL = 3
integer ::  Def_electronVolt_unitsFlag_CTRL = 4
integer ::  Def_Kelvin_unitsFlag_CTRL      = 5
integer ::  Def_Internal_unitsFlag_CTRL    = 6
integer ::  MX_char_unitsName_len = 10

public :: get_units_flag
public :: get_units_name

CONTAINS
subroutine get_units_flag(ch,unit_type)
 character(*), intent(IN) :: ch
 integer, intent(OUT) :: unit_type

    if (trim(ch) == 'kJ/mol'.or.trim(ch) == 'kj/mol'.or.trim(ch) == 'KJ/mol') then
      unit_type = Def_kJpermol_unitsFlag_CTRL
    elseif(trim(ch) == 'kCal/mol'.or.trim(ch) == 'kcal/mol'.or.trim(ch) == 'KCal/mol')then
      unit_type = Def_kcalpermol_unitsFlag_CTRL
    elseif(trim(ch) == 'a.u.'.or.trim(ch) == 'u.a.'.or.trim(ch)=='A.U.'.or.trim(ch)=='U.A.') then
      unit_type = Def_atomicUnits_unitsFlag_CTRL
    elseif(trim(ch) == 'ev'.or.trim(ch) == 'EV'.or.trim(ch) == 'eV')then
      unit_type = Def_electronVolt_unitsFlag_CTRL
    elseif(trim(ch)=='Kelvin'.or.trim(ch)=='kelvin'.or.trim(ch)=='KELVIN') then
      unit_type = Def_Kelvin_unitsFlag_CTRL
    elseif(trim(ch)=='i.u.'.or.trim(ch)=='u.i.'.or.trim(ch)=='I.U.'.or.trim(ch)=='U.I.') then
      unit_type = Def_Internal_unitsFlag_CTRL
    else
      unit_type = Def_kJpermol_unitsFlag_CTRL ! default kJ/mol
    endif
end subroutine get_units_flag

subroutine get_units_name(ch,unit_type)
 implicit none
 character(MX_char_unitsName_len), intent(OUT) :: ch
 integer, intent(IN) :: unit_type
     ch(:) = ' '
    if (unit_type == Def_kJpermol_unitsFlag_CTRL) then
      ch = 'kJ/mol'
    elseif(unit_type == Def_kcalpermol_unitsFlag_CTRL)then
      ch = 'kcal/mol'
    elseif( unit_type == Def_atomicUnits_unitsFlag_CTRL) then
       ch = 'a.u.'
    elseif(unit_type == Def_electronVolt_unitsFlag_CTRL)then
       ch = 'eV'
    elseif(unit_type == Def_Kelvin_unitsFlag_CTRL) then
       ch = 'kelvin'
    elseif(unit_type == Def_Internal_unitsFlag_CTRL) then
       ch = 'i.u.'
    else
       ch = 'kJ/mol'
    endif
end subroutine get_units_name
end module units_def


