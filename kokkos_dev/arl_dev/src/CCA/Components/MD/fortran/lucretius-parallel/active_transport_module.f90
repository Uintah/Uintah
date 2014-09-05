module active_transport_module
type active_transport_type
 logical :: do_it
 logical, allocatable :: applied_on_type_mol(:)
 integer, allocatable :: sign_on_type_mol(:) 
 integer conserve_MOM !-1 = NO ; 0 = distribute it uniform to all molecules ; 1 distrib uniform to other molecules 
 real(8) magnitude
end type active_transport_type
type(active_transport_type) active_transport
contains
subroutine default_active_transport
  active_transport%do_it = .false.
  magnitude = 0.001d0
end default_active_transport

subroutine parse_active_transport
!ACTIVE_TRANSPORT conserve_MOM -1  apply_on_type_mol ( 2 -1 3 1 )  magnitude 0.001
call default_active_transport 
call search_words(1,lines,lines,Max_words_per_line,&
                  the_words,SizeOfLine,NumberOfWords,&
                  'ACTIVE_TRANSPORT',l_skip_line,which,trim(nf),.false.)
   active_transport%do_it = which%find
   if (.not.active_transport%do_it) RETURN
    
end module active_transport_module
