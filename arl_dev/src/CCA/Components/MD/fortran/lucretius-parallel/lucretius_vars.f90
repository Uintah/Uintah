module thermostat_Lucretius_data
implicit none
        real(8) g_logv(3),x_logv(3),v_logv(3),g_logs(10),x_logs(10),v_logs(10)
        real(8) wd_t1(10),wd_t2(10),wd_t4(10),wd_t8(10)
        integer :: nhc_step1=5   
        integer :: nhc_step2=5
        logical :: use_Lucretius_integrator = .false.
        integer :: Multi_Med = 1
        integer :: Multi_Big = 1
        integer :: N_N_O_S = 1

       real(8),allocatable:: chain_v_logs_xx(:,:), chain_v_logs_yy(:,:),chain_v_logs_zz(:,:)
       real(8),allocatable:: chain_g_logs_xx(:,:), chain_g_logs_yy(:,:),chain_g_logs_zz(:,:)
       real(8),allocatable:: chain_v_logv_xx(:,:), chain_v_logv_yy(:,:),chain_v_logv_zz(:,:)
       real(8),allocatable:: chain_g_logv_xx(:,:), chain_g_logv_yy(:,:),chain_g_logv_zz(:,:)
       real(8),allocatable:: chain_logv_xx(:,:), chain_logv_yy(:,:), chain_logv_zz(:,:)
       real(8),allocatable:: chain_logs_xx(:,:), chain_logs_yy(:,:), chain_logs_zz(:,:)


contains
subroutine allocate_thermostat_Lucretius_data
use ALL_atoms_data, only : Natoms
implicit none
!if (use_Lucretius_integrator) then
!if (i_type_thermostat_CTRL==-99.or.i_type_thermostat_CTRL==-98.or.i_type_thermostat_CTRL==-97.or.i_type_thermostat_CTRL==-96)then
 allocate(chain_v_logs_xx(Natoms,N_N_O_S+1),chain_v_logs_yy(Natoms,N_N_O_S+1),chain_v_logs_zz(Natoms,N_N_O_S+1))
 chain_v_logs_xx=0.0d0;chain_v_logs_yy=0.0d0;chain_v_logs_zz=0.0d0
 allocate(chain_g_logs_xx(Natoms,N_N_O_S+1),chain_g_logs_yy(Natoms,N_N_O_S+1),chain_g_logs_zz(Natoms,N_N_O_S+1))
 chain_g_logs_xx=0.0d0;chain_g_logs_yy=0.0d0;chain_g_logs_zz=0.0d0
 allocate(chain_logv_xx(Natoms,N_N_O_S+1),chain_logv_yy(Natoms,N_N_O_S+1),chain_logv_zz(Natoms,N_N_O_S+1))
 chain_logv_xx=0.0d0;chain_logv_yy=0.0d0;chain_logv_zz=0.0d0
 allocate(chain_logs_xx(Natoms,N_N_O_S+1),chain_logs_yy(Natoms,N_N_O_S+1),chain_logs_zz(Natoms,N_N_O_S+1))
 chain_logs_xx=0.0d0;chain_logs_yy=0.0d0;chain_logs_zz=0.0d0
 allocate(chain_v_logv_xx(Natoms,N_N_O_S+1),chain_v_logv_yy(Natoms,N_N_O_S+1),chain_v_logv_zz(Natoms,N_N_O_S+1))
 chain_v_logv_xx=0.0d0;chain_v_logv_yy=0.0d0;chain_v_logv_zz=0.0d0
 allocate(chain_g_logv_xx(Natoms,N_N_O_S+1),chain_g_logv_yy(Natoms,N_N_O_S+1),chain_g_logv_zz(Natoms,N_N_O_S+1))
 chain_g_logv_xx=0.0d0;chain_g_logv_yy=0.0d0;chain_g_logv_zz=0.0d0
!endif 
!endif
end subroutine allocate_thermostat_Lucretius_data
end module thermostat_Lucretius_data

