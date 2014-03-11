module error_handler_module
integer :: error_code = 0
contains
subroutine error_handler(where_,message)
use comunications , only : COMM_exit
character(*) , intent(IN) :: message,where_

if (trim(where_)/='') print*, 'Error mesage in code while executing ', trim(where_)
if (trim(message)/='') print*,' additional info=',trim(message)

if (code_error==100) then
 print*, 'The maximum size of nonbonded list exceeds the maximum allowed'
 print*, 'To do: Increase the constant MAX_NEIGH and recompile and restart the code'
else
endif

call COMM_exit();

end subroutine error_handler
end module error_handler_module
