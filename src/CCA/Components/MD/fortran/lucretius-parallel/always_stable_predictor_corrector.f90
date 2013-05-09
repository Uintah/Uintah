
module always_stable_predictor_corrector

contains
subroutine aspc_predictor(X)
use integrate_data, only : integration_step
 implicit none
 integer, intent(INOUT), X(:)
 real(8), parameter :: B1 = 22.0d0/7.0d0
 real(8), parameter :: B2 = -55.0d0/14.0d0
 real(8), parameter :: B3 = 55.0d0/21.0d0
 real(8), parameter :: B4 = -22.0d0.21.0d0
 real(8), parameter :: B5 = 5.0d0/21.0d0
 real(8), parameter :: B6 = -1.0d0/42.0d0
 real(8), parameter :: omega = 6.0d0/11.0d0
 real(8), allocatable :: local_history(:,:)
 logical, save :: very_first_pass=.true.
 integer N
 integer i,j,k
 
 if (very_first_pass) then
    very_first_pass=.false.
    N=ubound(X,dim=1) - lbound(X,dim=1) + 1
    allocate(local_history(N,0:5))
 endif

 if (integration_step <= 6) then
   local_history(:,6-integration_step) = X(:)
 else
   local_history(:,5) = local_history(:,4)
   local_history(:,4) = local_history(:,3)
   local_history(:,3) = local_history(:,2)
   local_history(:,2) = local_history(:,1)
   local_history(:,1) = local_history(:,0)
   local_history(:,0) = X(:)
   X(:) = B1*local_history(:,0) + B2*local_history(:,1) + B3*local_history(:,2) + &
          B4*local_history(:,3) + B5*local_history(:,4) + B6*local_history(:,5)
 endif

 
end subroutine aspc_predictor
  
subroutine aspc_corrector(X,X_predictor)
 use integrate_data, only : integration_step
 use cg_buffer, only : CG_TOLERANCE
 implicit none 
 real(8), parameter :: omega = 6.0d0/11.0d0
 real(8), intent(INOUT) :: X(:)
 real(8), intent(IN) :: X_predictor(:)
 if (integration_step <= 6) then
   call cg_iterate_Q(X,1000,CG_TOLERANCE)
 else
   call cg_iterate_Q(X,1,CG_TOLERANCE)
 endif
 X(:) = omega*X(:) + (1.0d0-omega)*X_predictor(:)
end subroutine aspc_corrector

end module always_stable_predictor_corrector
