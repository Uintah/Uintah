

subroutine set_cg_flags
use cg_buffer, only : l_DO_CG_CTRL_Q,l_DO_CG_CTRL_DIP,cg_skip_MAIN,l_DO_CG_CTRL
use integrate_data, only : integration_step
use CTRLs_data, only : l_ANY_DIPOLE_POL_CTRL,l_ANY_SFIELD_CTRL,l_skip_cg_in_first_step_CTRL

implicit none

 l_DO_CG_CTRL_Q   = .false.
 l_DO_CG_CTRL_DIP = .false.

 if (cg_skip_MAIN%Q/=0)then
  if (mod(integration_step, cg_skip_MAIN%Q)==0) then
    l_DO_CG_CTRL_Q = l_ANY_SFIELD_CTRL
  endif
 else 
    l_DO_CG_CTRL_Q = l_ANY_SFIELD_CTRL
 endif

 if (cg_skip_MAIN%DIP/=0)then
  if (mod(integration_step, cg_skip_MAIN%DIP)==0) then
    l_DO_CG_CTRL_DIP = l_ANY_DIPOLE_POL_CTRL
  endif
 else
    l_DO_CG_CTRL_DIP = l_ANY_DIPOLE_POL_CTRL
 endif

 if (integration_step==1) then
   l_DO_CG_CTRL_Q = l_ANY_SFIELD_CTRL
   l_DO_CG_CTRL_DIP = l_ANY_DIPOLE_POL_CTRL
 if (l_skip_cg_in_first_step_CTRL) then
   l_DO_CG_CTRL_Q =.false.
   l_DO_CG_CTRL_DIP=.false.
 endif
 endif
 l_DO_CG_CTRL = l_DO_CG_CTRL_Q.or.l_DO_CG_CTRL_DIP 



end subroutine set_cg_flags

