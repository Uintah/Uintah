module ensamble_def_module
implicit none
public :: ensamble_def
contains
subroutine ensamble_def(key,i_type_ensamble,i_type_thermos,i_type_barr,l_error)
  character(*),intent(IN) :: key
  integer, intent(OUT) :: i_type_ensamble,i_type_thermos,i_type_barr
  logical, intent(OUT) :: l_error

  l_error=.false.
 select case (trim(key))
 case('NVE')
  i_type_ensamble = 0
  i_type_thermos=0
  i_type_barr=0
 case ('NVT-BER')
   i_type_ensamble = 1
   i_type_barr=0
   i_type_thermos=1
 case('NVT-NH')
   i_type_ensamble = 2
   i_type_barr=0
   i_type_thermos=2
 case('NVT-NHC')
   i_type_ensamble = 3
   i_type_barr=0
   i_type_thermos=3
 case('NVT-LANGEVIN')
   i_type_ensamble = 4
   i_type_barr=0
    i_type_thermos=4
 case ('NPT-BER-ISO')
   i_type_ensamble = 5
   i_type_thermos=1
   i_type_barr=1
 case ('NPT-BER-XYZ')
   i_type_ensamble = 6
   i_type_thermos=1
   i_type_barr=2
 case ('NPT-BER-Z')
   i_type_ensamble = 7
   i_type_thermos=1
   i_type_barr=3
 case('NPT-NH')
   i_type_ensamble = 8
   i_type_thermos=2
   i_type_barr=4
 case('NPT-NHC')
   i_type_ensamble = 9
   i_type_thermos=2
   i_type_barr=5
 case('NPT-LANGEVIN')
   i_type_ensamble = 10
   i_type_thermos=3
   i_type_barr=6
 case default
   l_error = .true.
 end select
 
end subroutine ensamble_def
end module ensamble_def_module
