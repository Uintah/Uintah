
module Lucretius_integrator_module

private :: nhcint
private :: short_step_nonbonded ! is in from 'short_nonbonded.f90'
public :: Lucretius_integrator
logical, private, save, allocatable :: l_proceed(:)
contains

 include 'short_step_nonbonded.f90'  
 include 'cut_and_shift_nonbonded.f90'

 SUBROUTINE Lucretius_integrator
 use thermostat_Lucretius_data
 use ensamble_data
 use all_atoms_data, only : vxx,vyy,vzz,Natoms,fxx,fyy,fzz, is_dummy, all_atoms_mass,xxx,yyy,zzz,l_WALL_CTRL,&
                            fshort_xx,fshort_yy,fshort_zz, all_atoms_massinv,&
                            any_intramol_constrain_per_atom
 use sizes_data, only : Ndummies,Nconstrains
 use intramolecular_forces
 use force_driver_module
 use dummy_module 
 use stresses_data, only : stress, stress_dummy
 use integrate_data, only : time_step, integration_step, lucretius_integrator_more_speed_doit, lucretius_integrator_more_speed_skip
 use kinetics, only : get_kinetic_energy_stress, get_instant_temperature, is_list_2BE_updated, add_kinetic_pressure
 use non_bonded_lists_data, only : l_update_VERLET_LIST
 use sim_cel_data, only : sim_cel
 use intramol_constrains, only : shake_ll
 use boundaries
 use thetering_forces_module
 use thetering_data, only : thetering

  implicit none    
  real(8), save :: time_step_short, dM, half_time_step
  logical, save :: l_very_first = .true.
  logical, save :: nve, nvt, npt, nvtchains
  integer i,j,k,istep,iat
  logical go_in_inner_2_cycle, go_in_inner_3_cycle
  real(8), allocatable,save  :: fnow_xx(:),fnow_yy(:),fnow_zz(:), buff(:)
  real(8), allocatable :: fref_xx(:),fref_yy(:),fref_zz(:)
  real(8), allocatable :: xxx1(:),yyy1(:),zzz1(:), xxx2(:),yyy2(:),zzz2(:)
  real(8), allocatable :: xxx_shaken(:),yyy_shaken(:),zzz_shaken(:)
  real(8) tmp, t(3),d,idt,dt2
  real(8), save :: e_2,e_4,e_6,e_8 ! parameters for barostat integration

 allocate(fref_xx(Natoms),fref_yy(Natoms),fref_zz(Natoms))

!write(14,*)Natoms
!do i = 1, Natoms
!write(14,'(I7,1X,3(F16.7,1X))') i,fxx(i),fyy(i),fzz(i)
!enddo
!STOP

 if (l_very_first) then
    l_very_first = .false.
    call very_first_pass
 end if

 call initialize
  d = dble(Multi_Med)
  dt2 = (time_step_short*time_step_short)
  idt = 1.0d0/dt2

  do istep = 1,Multi_Big
    where(l_proceed)
      vxx(:) = vxx(:) + buff(:)*fnow_xx(:)
      vyy(:) = vyy(:) + buff(:)*fnow_yy(:)
      vzz(:) = vzz(:) + buff(:)*fnow_zz(:)
    endwhere
    call update_innermost_distances
    call is_list_2BE_updated(l_update_VERLET_LIST)
    if (Ndummies>0) then 
          call Do_Dummy_Coords()   ! update positions of dummy atoms
          call is_list_2BE_updated(l_update_VERLET_LIST)
    endif
      fxx(:) = 0.0d0; fyy(:) = 0.0d0 ; fzz(:) = 0.0d0
      fshort_xx(:)=0.0d0 ; 
      fshort_yy(:)=0.0d0 ; 
      fshort_zz(:)= 0.0d0
      stress(:) = 0.0d0
      call inner_1_SHELL_forces
      fref_xx(:) = fxx(:); fref_yy(:) = fyy(:); fref_zz(:) = fzz(:)
      fnow_xx(:) = fxx(:); fnow_yy(:) = fyy(:); fnow_zz(:) = fzz(:)
      go_in_inner_2_cycle = (mod(istep,Multi_Med)==0).and.(istep/=Multi_Big)
      if (go_in_inner_2_cycle) then
          call inner_2_SHELL_forces
          fnow_xx(:) = fref_xx(:) + d*(fxx(:)-fref_xx(:))
          fnow_yy(:) = fref_yy(:) + d*(fyy(:)-fref_yy(:))
          fnow_zz(:) = fref_zz(:) + d*(fzz(:)-fref_zz(:))
      endif
        go_in_inner_3_cycle = istep == Multi_Big
        if (go_in_inner_3_cycle)then
             fxx(:)=0.0d0; fyy(:)=0.0d0; fzz(:) = 0.0d0
             call dihedral_forces       !en4cen()
             fshort_xx(:) = fref_xx(:) + fxx(:)
             fshort_yy(:) = fref_yy(:) + fyy(:)
             fshort_zz(:) = fref_zz(:) + fzz(:)
             fxx(:) = fxx(:) + fref_xx(:)
             fyy(:) = fyy(:) + fref_yy(:)
             fzz(:) = fzz(:) + fref_zz(:)
             if (lucretius_integrator_more_speed_doit.or.lucretius_integrator_more_speed_skip > 1) then  ! use it when prepare systems; 
              if (mod(integration_step,lucretius_integrator_more_speed_skip)/=0) then
              call cut_and_shift_nonbonded   ! I do not have yet dipoles in
              else
              call force_driver
              endif
             else
               call force_driver
             endif
             if (Ndummies>0) then ! dummy forces needs to be done after finishing intramolecular part
                call Do_Dummy_Forces(.false.,fshort_xx,fshort_yy,fshort_zz)
                call Do_Dummy_Forces(.true.,fxx,fyy,fzz)
              endif
              
!             stress(:) = stress(:) + stress_dummy(:)
             fnow_xx(:) = fref_xx(:) + d*(fshort_xx(:)-fref_xx(:))
             fnow_yy(:) = fref_yy(:) + d*(fshort_yy(:)-fref_yy(:))
             fnow_zz(:) = fref_zz(:) + d*(fshort_zz(:)-fref_zz(:))
            if (Nconstrains > 0)then
            where (any_intramol_constrain_per_atom)
              xxx1 = xxx + time_step_short*vxx + dt2*fxx*all_atoms_massinv
              yyy1 = yyy + time_step_short*vyy + dt2*fyy*all_atoms_massinv
              zzz1 = zzz + time_step_short*vzz + dt2*fzz*all_atoms_massinv
            endwhere
            call shake_ll(.true.,xxx1,yyy1,zzz1,xxx_shaken,yyy_shaken,zzz_shaken)
            end if
           
             fnow_xx(:) = fnow_xx(:) + dM*(fxx(:)-fshort_xx(:))
             fnow_yy(:) = fnow_yy(:) + dM*(fyy(:)-fshort_yy(:))
             fnow_zz(:) = fnow_zz(:) + dM*(fzz(:)-fshort_zz(:))

       end if   ! go_in_inner_3_cycle
      if (Nconstrains>0)then
        where (any_intramol_constrain_per_atom)
         xxx1 = xxx + time_step_short*vxx + dt2*fnow_xx*all_atoms_massinv
         yyy1 = yyy + time_step_short*vyy + dt2*fnow_yy*all_atoms_massinv
         zzz1 = zzz + time_step_short*vzz + dt2*fnow_zz*all_atoms_massinv
        endwhere
        call shake_ll(.false.,xxx1,yyy1,zzz1,xxx_shaken,yyy_shaken,zzz_shaken)
        where (any_intramol_constrain_per_atom)
          xxx2 = xxx_shaken-xxx
          yyy2 = yyy_shaken-yyy
          zzz2 = zzz_shaken-zzz
        endwhere 
        call periodic_images(xxx2,yyy2,zzz2)
        where (any_intramol_constrain_per_atom)
         fnow_xx = (xxx2-time_step_short*vxx)*all_atoms_mass * idt
         fnow_yy = (yyy2-time_step_short*vyy)*all_atoms_mass * idt
         fnow_zz = (zzz2-time_step_short*vzz)*all_atoms_mass * idt
        endwhere
      endif
      
    where(l_proceed)
      vxx(:) = vxx(:) + buff(:)*fnow_xx(:)
      vyy(:) = vyy(:) + buff(:)*fnow_yy(:)
      vzz(:) = vzz(:) + buff(:)*fnow_zz(:)
    endwhere


     enddo ! istep

     call local_thermos
     if (npt) call nptint
     call get_kinetic_energy_stress(T_eval)
     call add_kinetic_pressure

!open(unit=14,file='fort.14',recl=300)
!do i = 1, Natoms
!write(14,*)i,vxx(i)*100.0d0,vyy(i)*100.0d0,vzz(i)*100.0d0
!enddo
!stop

    
   deallocate(fref_xx,fref_yy,fref_zz)
      if (Nconstrains > 0) then
        deallocate(xxx1,yyy1,zzz1)
        deallocate(xxx_shaken,yyy_shaken,zzz_shaken)
        deallocate(xxx2,yyy2,zzz2)
      endif

   contains
   
    subroutine initialize
      if (Nconstrains > 0) then
        allocate(xxx1(Natoms),yyy1(Natoms),zzz1(Natoms))
        allocate(xxx_shaken(Natoms),yyy_shaken(Natoms),zzz_shaken(Natoms))
        allocate(xxx2(Natoms),yyy2(Natoms),zzz2(Natoms))
      endif
      call local_thermos;
      if (npt) call nptint
      call get_kinetic_energy_stress(T_eval)
    end  subroutine initialize
    subroutine local_thermos
      if (nvt) then
          call nhcint
      else  if (nvtchains) then
        if (i_type_thermostat_CTRL == -99) then
          call nhcint_masive_chains_atom_xyz
        else if (i_type_thermostat_CTRL == -98) then
          call nhcint_masive_chains_atom
        else if (i_type_thermostat_CTRL == -97) then
          call nhcint_masive_chains_mol_xyz
        else if (i_type_thermostat_CTRL == -96) then
          call nhcint_masive_chains_mol
        endif
      endif
    end  subroutine local_thermos
   
    subroutine inner_1_SHELL_forces  !innermost  
           call bond_forces             !en2cen()
           call angle_forces            !en3cen()
           call out_of_plane_deforms    !improper()
           call thetering_forces
    end subroutine inner_1_SHELL_forces
        
    subroutine inner_2_SHELL_forces    
           call dihedral_forces               !en4cen()                
           call short_step_nonbonded(fxx,fyy,fzz)          !interchs()
           if (Ndummies>0)  call Do_Dummy_Forces(.false.,fxx,fyy,fzz)
    end subroutine inner_2_SHELL_forces
   


subroutine update_innermost_distances
integer kk
real(8) aa,bb,aa2,scala,arg2,poly

    if (nve.or.nvt.or.nvtchains)then
    where(l_proceed)
           xxx(:) = xxx(:) + time_step_short * vxx(:)
           yyy(:) = yyy(:) + time_step_short * vyy(:)
           zzz(:) = zzz(:) + time_step_short * vzz(:)
    endwhere
    endif
    if (npt) then
          do kk = 1,3
            aa = dexp(half_time_step*v_logv(kk))
            aa2 = aa*aa
            arg2 = (v_logv(kk)*half_time_step)*(v_logv(kk)*half_time_step)
            poly = (((e_8*arg2+e_6)*arg2+e_4)*arg2+e_2)*arg2+1.d0
            bb = aa*poly*time_step_short
            xxx(:) = xxx(:)*aa2 + vxx(:)*bb
            yyy(:) = yyy(:)*aa2 + vyy(:)*bb
            zzz(:) = zzz(:)*aa2 + vzz(:)*bb
            x_logv(kk) = x_logv(kk) + v_logv(kk)*time_step_short
            scala = dexp(v_logv(kk)*time_step_short)
            sim_cel(1)=sim_cel(1)*scala ! This is for ortho case only
            sim_cel(5)=sim_cel(5)*scala
            sim_cel(9)=sim_cel(9)*scala
          end do
     end if
     
end subroutine update_innermost_distances

 
    subroutine very_first_pass
      real(8) wys(5) 
      integer o
      allocate(fnow_xx(Natoms),fnow_yy(Natoms),fnow_zz(Natoms))
      fnow_xx=fxx ; fnow_yy=fyy; fnow_zz=fzz

      allocate(l_proceed(Natoms))
      do i = 1, Natoms
        l_proceed(i) = .not.(is_dummy(i).or.l_WALL_CTRL(i).or.all_atoms_mass(i)<1.0d-8)
      enddo

      dM=dble(Multi_Big)
      time_step_short = time_step/dM
      half_time_step = time_step_short * 0.5d0
      allocate(buff(Natoms))
      where(l_proceed)
        buff(:) = half_time_step / all_atoms_mass(:)
      endwhere


       fshort_xx=0.0d0; fshort_yy=0.0d0; fshort_zz=0.0d0
!     *****parameters in the Yoshida/Suzuki integration
      wys(1) = 1.d0
      if (nhc_step2==5) then
        wys(1) = 1.d0/(4.d0-4.d0**(1.d0/3.d0))
        wys(2) = 1.d0/(4.d0-4.d0**(1.d0/3.d0))
        wys(3) =-4.d0**(1.d0/3.d0)/(4.d0-4.d0**(1.d0/3.d0))
        wys(4) = 1.d0/(4.d0-4.d0**(1.d0/3.d0))
        wys(5) = 1.d0/(4.d0-4.d0**(1.d0/3.d0))
      end if
      if (nhc_step2==3) then
        wys(1) = 1.d0/(2.d0-2.d0**(1.d0/3.d0))
        wys(2) =-2.d0**(1.d0/3.d0)/(2.d0-2.d0**(1.d0/3.d0))
        wys(3) = 1.d0/(2.d0-2.d0**(1.d0/3.d0))
      end if
      do i = 1,nhc_step2
        wd_t1(i) = wys(i)/dble(nhc_step1)  * dble(Multi_Med)
        wd_t2(i) = wd_t1(i)/2.0d0
        wd_t4(i) = wd_t1(i)/4.0d0
        wd_t8(i) = wd_t1(i)/8.0d0
      end do
!print*,'wys = ',wys(:)
!print*,'nhcstep1 nhcstep2=',nhc_step1,nhc_step2
!print*,'wdt1=',wd_t1(:)
!print*,'wdt2=',wd_t2(:)
!print*,'wdt4=',wd_t4(:)
!print*,'wdt8=',wd_t8(:)
!read(*,*)
     g_logv=0.0d0; x_logv=0.0d0; v_logv=0.0d0; 
     g_logs=0.0d0; x_logs=0.0d0; v_logs=0.0d0
     nvtchains=i_type_thermostat_CTRL == -99.or.i_type_thermostat_CTRL == -98.or.i_type_thermostat_CTRL == -97.or.i_type_thermostat_CTRL == -96 
     nvt = i_type_thermostat_CTRL > 0
     npt = i_type_barostat_CTRL > 0
     nve = i_type_ensamble_CTRL == 0 .and. (.not.nvtchains) .and. (.not.nvt) .and. (.not.npt)

     if (nvtchains) then
    call allocate_thermostat_Lucretius_data
    chain_g_logv_xx=0.0d0; chain_g_logv_yy=0.0d0; chain_g_logv_zz=0.0d0
    chain_g_logs_xx=0.0d0; chain_g_logs_yy=0.0d0; chain_g_logs_zz=0.0d0
    chain_v_logv_xx=0.0d0; chain_v_logv_yy=0.0d0; chain_v_logv_zz=0.0d0
    chain_v_logs_xx=0.0d0; chain_v_logs_yy=0.0d0; chain_v_logs_zz=0.0d0
    chain_logs_xx  =0.0d0; chain_logs_yy  =0.0d0; chain_logs_zz  =0.0d0
    chain_logv_xx  =0.0d0; chain_logv_yy  =0.0d0; chain_logv_zz  =0.0d0 

     endif

     if (npt) nvt=.false.
     if (nve.and.npt) then
       print*, 'ERROR with nve npt flags in integrate lucretius ; they are both true ; Choose one but not both at the same time'
       STOP
     endif
     if (nve.and.nvt) then
       print*, 'ERROR with nve nvt flags in integrate lucretius ; they are both true ; Choose one but not both at the same time'
       STOP
     endif
     if ((.not.nve).and.(.not.nvt).and.(.not.nvtchains)) then
       print*, 'ERROR with nve nvt nvtchains flags in integrate lucretius ; they are all FALSE ; Just choose one of the,'
       STOP
     endif
  
print*,'in lucretius integrator : nve nvt npt nvtchains ',nve,nvt,npt,nvtchains
 
      e_2 = 1.d0/6.d0  !parameters for yoshida suzuki integration of barostst
      e_4 = e_2/20.d0
      e_6 = e_4/42.d0
      e_8 = e_6/72.d0
    end  subroutine very_first_pass
        
 end subroutine Lucretius_integrator
      
      
  subroutine nhcint  ! Make it a private object
      use kinetics, only : get_instant_temperature, add_kinetic_pressure
      use thermostat_Lucretius_data
      use ALL_atoms_data, only : Natoms, vxx,vyy,vzz
      use integrate_data, only : time_step
      use ensamble_data, only : thermo_coupling, T_eval, temperature
      use DOF_data
      integer i,j,inos,iat,kk
      real(8) csi,aa, tau2,fct
      real(8) T_imposed

      T_imposed = temperature

      call get_instant_temperature(T_eval)
      call add_kinetic_pressure

!print*, 'entered in nhcint with T imposed eval=',T_imposed,T_eval

      tau2 = (thermo_coupling/time_step)**2
!print*, 'thermo_coupling time_step=',thermo_coupling,time_step
!print*, 'tau2=',tau2,thermo_coupling, v_logs(1)
      csi = 1.d0
      g_logs(1) = (T_eval/T_imposed - 1.0d0)/tau2
!print*, 'glogs(1) = ',g_logs(1), 'tau2=',tau2
!read(*,*)
      do i = 1,nhc_step1
        do j = 1,nhc_step2
          v_logs(N_N_O_S) = v_logs(N_N_O_S)+g_logs(N_N_O_S)*wd_t4(j)
          do inos = 1, N_N_O_S-1
            aa = dexp(-wd_t8(j)*v_logs(N_N_O_S+1-inos))
            v_logs(N_N_O_S-inos) = v_logs(N_N_O_S-inos)*aa*aa + wd_t4(j)*g_logs(N_N_O_S-inos)*aa
          end do
          aa = dexp(-wd_t2(j)*v_logs(1))
          csi = csi*aa
!print*,'v_logs g_logs wd_t4=',v_logs(N_N_O_S),g_logs(N_N_O_S),wd_t4(j)
!print*,'i j aa csi',i,j,aa,csi
!read(*,*)
          g_logs(1) = (csi*csi*T_eval/T_imposed - 1.0d0)/tau2                    !(csi*csi*ekin2 - gnkt)/qtmass(1)     
          x_logs(1:N_N_O_S) = x_logs(1:N_N_O_S) + v_logs(1:N_N_O_S)* wd_t2(j)
          do inos = 1, N_N_O_S-1
            if (inos==1) then 
               fct = DOF_total ! it could be incorrect here
            else
               fct = 1.0d0
            endif
            aa = dexp(-wd_t8(j)*v_logs(inos+1)) + wd_t4(j)*g_logs(inos)*aa
            g_logs(inos+1) = v_logs(inos)*v_logs(inos) *  fct - 1.0d0/tau2   ! I think it is not correct for NNOS > 1; JUST USE NNOS = 1 
          end do
          v_logs(N_N_O_S) = v_logs(N_N_O_S) + g_logs(N_N_O_S)*wd_t4(j)
        end do
      end do

!print*, 'csi=',csi
!stop
      vxx(:) = vxx(:) * csi
      vyy(:) = vyy(:) * csi
      vzz(:) = vzz(:) * csi
!print*,'----csi=',csi,T_eval
!read(*,*)

   end subroutine nhcint
      


  subroutine nhcint_masive_chains_atom_xyz  ! Make it a private object
      use kinetics, only : get_instant_temperature, add_kinetic_pressure
      use thermostat_Lucretius_data
      use ALL_atoms_data, only : Natoms, vxx,vyy,vzz, all_atoms_mass,atom_dof
      use integrate_data, only : time_step
      use ensamble_data, only : thermo_coupling, T_eval, temperature
      use DOF_data
      use physical_constants, only : Red_Boltzmann_constant
      integer i,j,inos,iat,kk
      real(8) csi,aa, tau2,fct
      real(8) T_imposed , Tx(Natoms),Ty(Natoms),Tz(Natoms)

      T_imposed = temperature

      call get_instant_temperature(T_eval)
      call add_kinetic_pressure
!print*, 'entered in nhcint with T imposed eval=',T_imposed,T_eval

      tau2 = (thermo_coupling/time_step)**2

      ff = 3.0d0 / Red_Boltzmann_constant
      
      do iat = 1, Natoms
      csi_xx = 1.0d0
      csi_yy = 1.0d0
      csi_zz = 1.0d0
!      if ((.not.is_WALL(iat)).and.(.not.is_dummy(iat)).and.(atom_dof(iat)>0) ) then
      if (l_proceed(iat)) then
      c =  all_atoms_mass(iat)  /   atom_dof(iat) * ff 
      Tx(iat) = c * vxx(iat)**2; Ty(iat) = c * vyy(iat)**2 ; Tz(iat) = c * vzz(iat)**2 
      chain_g_logs_xx(iat,1) = (Tx(iat)/T_imposed - 1.0d0)/tau2  
      chain_g_logs_yy(iat,1) = (Ty(iat)/T_imposed - 1.0d0)/tau2    
      chain_g_logs_zz(iat,1) = (Tz(iat)/T_imposed - 1.0d0)/tau2                        
      do i = 1,nhc_step1
        do j = 1,nhc_step2
          chain_v_logs_xx(iat,N_N_O_S) = chain_v_logs_xx(iat,N_N_O_S)+chain_g_logs_xx(iat,N_N_O_S)*wd_t4(j)
          chain_v_logs_yy(iat,N_N_O_S) = chain_v_logs_yy(iat,N_N_O_S)+chain_g_logs_yy(iat,N_N_O_S)*wd_t4(j)
          chain_v_logs_zz(iat,N_N_O_S) = chain_v_logs_zz(iat,N_N_O_S)+chain_g_logs_zz(iat,N_N_O_S)*wd_t4(j)
          do inos = 1, N_N_O_S-1
            aa = dexp(-wd_t8(j)*chain_v_logs_xx(iat,N_N_O_S+1-inos))
            chain_v_logs_xx(iat,N_N_O_S-inos) = chain_v_logs_xx(iat,N_N_O_S-inos)*aa*aa + wd_t4(j)*chain_g_logs_xx(iat,N_N_O_S-inos)*aa
            aa = dexp(-wd_t8(j)*chain_v_logs_yy(iat,N_N_O_S+1-inos))
            chain_v_logs_yy(iat,N_N_O_S-inos) = chain_v_logs_yy(iat,N_N_O_S-inos)*aa*aa + wd_t4(j)*chain_g_logs_yy(iat,N_N_O_S-inos)*aa
            aa = dexp(-wd_t8(j)*chain_v_logs_zz(iat,N_N_O_S+1-inos))
            chain_v_logs_zz(iat,N_N_O_S-inos) = chain_v_logs_zz(iat,N_N_O_S-inos)*aa*aa + wd_t4(j)*chain_g_logs_zz(iat,N_N_O_S-inos)*aa                        
          end do
          aa_xx = dexp(-wd_t2(j)*chain_v_logs_xx(iat,1)); 
          aa_yy = dexp(-wd_t2(j)*chain_v_logs_yy(iat,1));           
          aa_zz = dexp(-wd_t2(j)*chain_v_logs_zz(iat,1)); 
          csi_xx = csi_xx*aa_xx
          csi_yy = csi_yy*aa_yy 
          csi_zz = csi_zz*aa_zz
          chain_g_logs_xx(iat,1) = (csi_xx*csi_xx*Tx(iat)/T_imposed - 1.0d0)/tau2   
          chain_g_logs_yy(iat,1) = (csi_yy*csi_yy*Ty(iat)/T_imposed - 1.0d0)/tau2       
          chain_g_logs_zz(iat,1) = (csi_zz*csi_zz*Tz(iat)/T_imposed - 1.0d0)/tau2                        
          chain_logs_xx(iat,1:N_N_O_S) = chain_logs_xx(iat,1:N_N_O_S) + chain_v_logs_xx(iat,1:N_N_O_S)* wd_t2(j)
          chain_logs_yy(iat,1:N_N_O_S) = chain_logs_yy(iat,1:N_N_O_S) + chain_v_logs_yy(iat,1:N_N_O_S)* wd_t2(j)
          chain_logs_zz(iat,1:N_N_O_S) = chain_logs_zz(iat,1:N_N_O_S) + chain_v_logs_zz(iat,1:N_N_O_S)* wd_t2(j) 
          do inos = 1, N_N_O_S-1
            if (inos==1) then
               fct = (atom_dof(iat)/3.0d0) !DOF_total ! it could be incorrect here
            else
               fct = 1.0d0
            endif
            aa_xx = dexp(-wd_t8(j)*chain_v_logs_xx(iat,inos+1)) + wd_t4(j)*chain_g_logs_xx(iat,inos)*aa_xx
            aa_yy = dexp(-wd_t8(j)*chain_v_logs_yy(iat,inos+1)) + wd_t4(j)*chain_g_logs_yy(iat,inos)*aa_yy
            aa_zz = dexp(-wd_t8(j)*chain_v_logs_zz(iat,inos+1)) + wd_t4(j)*chain_g_logs_zz(iat,inos)*aa_zz
            chain_g_logs_xx(iat,inos+1) = chain_v_logs_xx(iat,inos)*chain_v_logs_xx(iat,inos) *  fct - 1.0d0/tau2 
            chain_g_logs_yy(iat,inos+1) = chain_v_logs_yy(iat,inos)*chain_v_logs_yy(iat,inos) *  fct - 1.0d0/tau2
            chain_g_logs_zz(iat,inos+1) = chain_v_logs_zz(iat,inos)*chain_v_logs_zz(iat,inos) *  fct - 1.0d0/tau2  
          end do
          chain_v_logs_xx(iat,N_N_O_S) = chain_v_logs_xx(iat,N_N_O_S) + chain_g_logs_xx(iat,N_N_O_S)*wd_t4(j)
          chain_v_logs_yy(iat,N_N_O_S) = chain_v_logs_yy(iat,N_N_O_S) + chain_g_logs_yy(iat,N_N_O_S)*wd_t4(j)
          chain_v_logs_zz(iat,N_N_O_S) = chain_v_logs_zz(iat,N_N_O_S) + chain_g_logs_zz(iat,N_N_O_S)*wd_t4(j)
        end do
       enddo
      vxx(iat) = vxx(iat) * csi_xx
      vyy(iat) = vyy(iat) * csi_yy
      vzz(iat) = vzz(iat) * csi_zz
  endif !  if ((.not.is_WALL(iat)).and.(.not.is_dummy(iat)).and.(atom_dof(iat)>0) )
  enddo !i  = 1, Natoms

   end subroutine nhcint_masive_chains_atom_xyz

   subroutine nhcint_masive_chains_atom
     use kinetics, only : get_instant_temperature, add_kinetic_pressure
      use thermostat_Lucretius_data
      use ALL_atoms_data, only : Natoms, vxx,vyy,vzz, all_atoms_mass,atom_dof
      use integrate_data, only : time_step
      use ensamble_data, only : thermo_coupling, T_eval, temperature
      use DOF_data
      use physical_constants, only : Red_Boltzmann_constant
      integer i,j,inos,iat,kk
      real(8) csi,aa, tau2,fct
      real(8) T_imposed , Tx(Natoms),Ty(Natoms),Tz(Natoms)
                                
      T_imposed = temperature
      call get_instant_temperature(T_eval)
      call add_kinetic_pressure

      tau2 = (thermo_coupling/time_step)**2
      ff = 1.0d0 / Red_Boltzmann_constant

      do iat = 1, Natoms
      csi_xx = 1.0d0
!      if ((.not.is_WALL(iat)).and.(.not.is_dummy(iat)).and.(atom_dof(iat)>0) ) then
      if (l_proceed(iat)) then
      c =  all_atoms_mass(iat)  /   atom_dof(iat) * ff
      Tx(iat) = c * (vxx(iat)**2 + vyy(iat)**2 + vzz(iat)**2 ) 
      chain_g_logs_xx(iat,1) = (Tx(iat)/T_imposed - 1.0d0)/tau2
      do i = 1,nhc_step1
        do j = 1,nhc_step2
          chain_v_logs_xx(iat,N_N_O_S) = chain_v_logs_xx(iat,N_N_O_S)+chain_g_logs_xx(iat,N_N_O_S)*wd_t4(j)
          do inos = 1, N_N_O_S-1
            aa = dexp(-wd_t8(j)*chain_v_logs_xx(iat,N_N_O_S+1-inos))
            chain_v_logs_xx(iat,N_N_O_S-inos) = chain_v_logs_xx(iat,N_N_O_S-inos)*aa*aa + wd_t4(j)*chain_g_logs_xx(iat,N_N_O_S-inos)*aa
          end do
          aa_xx = dexp(-wd_t2(j)*chain_v_logs_xx(iat,1));
          csi_xx = csi_xx*aa_xx
          chain_g_logs_xx(iat,1) = (csi_xx*csi_xx*Tx(iat)/T_imposed - 1.0d0)/tau2
          chain_logs_xx(iat,1:N_N_O_S) = chain_logs_xx(iat,1:N_N_O_S) + chain_v_logs_xx(iat,1:N_N_O_S)* wd_t2(j)
          do inos = 1, N_N_O_S-1
            if (inos==1) then
               fct = atom_dof(iat) !DOF_total ! it could be incorrect here
            else
               fct = 1.0d0
            endif
            aa_xx = dexp(-wd_t8(j)*chain_v_logs_xx(iat,inos+1)) + wd_t4(j)*chain_g_logs_xx(iat,inos)*aa_xx
            chain_g_logs_xx(iat,inos+1) = chain_v_logs_xx(iat,inos)*chain_v_logs_xx(iat,inos) *  fct - 1.0d0/tau2
          end do
          chain_v_logs_xx(iat,N_N_O_S) = chain_v_logs_xx(iat,N_N_O_S) + chain_g_logs_xx(iat,N_N_O_S)*wd_t4(j)
        end do
       enddo
!print*,iat,csi_xx,chain_g_logs_xx(iat,:)
      vxx(iat) = vxx(iat) * csi_xx
      vyy(iat) = vyy(iat) * csi_xx
      vzz(iat) = vzz(iat) * csi_xx
  endif !  if ((.not.is_WALL(iat)).and.(.not.is_dummy(iat)).and.(atom_dof(iat)>0) )
  enddo !i  = 1, Natoms
   end subroutine nhcint_masive_chains_atom

   subroutine nhcint_masive_chains_mol_xyz

      use kinetics, only : get_instant_temperature, add_kinetic_pressure
      use thermostat_Lucretius_data
      use ALL_atoms_data, only : Natoms, vxx,vyy,vzz, all_atoms_mass,atom_dof
      use ALL_mols_data, only :  Nmols, mol_dof,start_group,end_group
      use integrate_data, only : time_step
      use ensamble_data, only : thermo_coupling, T_eval, temperature
      use DOF_data
      use physical_constants, only : Red_Boltzmann_constant
      integer i,j,inos,iat,kk
      real(8) csi,aa, tau2,fct  ,s_dof, s_2kin_xx,s_2kin_yy,s_2kin_zz 
      real(8) T_imposed , Tx(Nmols),Ty(Nmols),Tz(Nmols)
                                
      T_imposed = temperature

      call get_instant_temperature(T_eval)
      call add_kinetic_pressure

      tau2 = (thermo_coupling/time_step)**2
      ff = 3.0d0 / Red_Boltzmann_constant

      do imol = 1, Nmols
      csi_xx = 1.0d0
      csi_yy = 1.0d0
      csi_zz = 1.0d0
      if (mol_dof(imol)>0) then
      s_2kin_xx = 0.0d0 
      s_2kin_yy = 0.0d0
      s_2kin_zz = 0.0d0
      do iat = start_group(imol),end_group(imol)      
      if (l_proceed(iat)) then
        s_2kin_xx = s_2kin_xx + all_atoms_mass(iat)*vxx(iat)**2 
        s_2kin_yy = s_2kin_yy + all_atoms_mass(iat)*vyy(iat)**2 
        s_2kin_zz = s_2kin_zz + all_atoms_mass(iat)*vzz(iat)**2   
      endif
      enddo
      Tx(imol) =  s_2kin_xx / mol_dof(imol) * ff
      Ty(imol) =  s_2kin_yy / mol_dof(imol) * ff
      Tz(imol) =  s_2kin_zz / mol_dof(imol) * ff
        
      chain_g_logs_xx(imol,1) = (Tx(imol)/T_imposed - 1.0d0)/tau2
      chain_g_logs_yy(imol,1) = (Ty(imol)/T_imposed - 1.0d0)/tau2
      chain_g_logs_zz(imol,1) = (Tz(imol)/T_imposed - 1.0d0)/tau2
      do i = 1,nhc_step1
        do j = 1,nhc_step2
          chain_v_logs_xx(imol,N_N_O_S) = chain_v_logs_xx(imol,N_N_O_S)+chain_g_logs_xx(imol,N_N_O_S)*wd_t4(j)
          chain_v_logs_yy(imol,N_N_O_S) = chain_v_logs_yy(imol,N_N_O_S)+chain_g_logs_yy(imol,N_N_O_S)*wd_t4(j) 
          chain_v_logs_zz(imol,N_N_O_S) = chain_v_logs_zz(imol,N_N_O_S)+chain_g_logs_zz(imol,N_N_O_S)*wd_t4(j)
          do inos = 1, N_N_O_S-1
            aa = dexp(-wd_t8(j)*chain_v_logs_xx(imol,N_N_O_S+1-inos))
            chain_v_logs_xx(imol,N_N_O_S-inos) = chain_v_logs_xx(imol,N_N_O_S-inos)*aa*aa + wd_t4(j)*chain_g_logs_xx(imol,N_N_O_S-inos)*aa
            aa = dexp(-wd_t8(j)*chain_v_logs_yy(imol,N_N_O_S+1-inos))
            chain_v_logs_yy(imol,N_N_O_S-inos) = chain_v_logs_yy(imol,N_N_O_S-inos)*aa*aa + wd_t4(j)*chain_g_logs_yy(imol,N_N_O_S-inos)*aa
            aa = dexp(-wd_t8(j)*chain_v_logs_zz(imol,N_N_O_S+1-inos))
            chain_v_logs_zz(imol,N_N_O_S-inos) = chain_v_logs_zz(imol,N_N_O_S-inos)*aa*aa + wd_t4(j)*chain_g_logs_zz(imol,N_N_O_S-inos)*aa
          end do
          aa_xx = dexp(-wd_t2(j)*chain_v_logs_xx(imol,1));
          aa_yy = dexp(-wd_t2(j)*chain_v_logs_yy(imol,1));
          aa_zz = dexp(-wd_t2(j)*chain_v_logs_zz(imol,1));
          csi_xx = csi_xx*aa_xx
          csi_yy = csi_yy*aa_yy
          csi_zz = csi_zz*aa_zz
          chain_g_logs_xx(imol,1) = (csi_xx*csi_xx*Tx(imol)/T_imposed - 1.0d0)/tau2
          chain_g_logs_yy(imol,1) = (csi_yy*csi_yy*Ty(imol)/T_imposed - 1.0d0)/tau2
          chain_g_logs_zz(imol,1) = (csi_zz*csi_zz*Tz(imol)/T_imposed - 1.0d0)/tau2
          chain_logs_xx(imol,1:N_N_O_S) = chain_logs_xx(imol,1:N_N_O_S) + chain_v_logs_xx(imol,1:N_N_O_S)* wd_t2(j)
          chain_logs_yy(imol,1:N_N_O_S) = chain_logs_yy(imol,1:N_N_O_S) + chain_v_logs_yy(imol,1:N_N_O_S)* wd_t2(j) 
          chain_logs_zz(imol,1:N_N_O_S) = chain_logs_zz(imol,1:N_N_O_S) + chain_v_logs_zz(imol,1:N_N_O_S)* wd_t2(j)
          do inos = 1, N_N_O_S-1
            if (inos==1) then
               fct = (mol_dof(imol)/3.0d0) !DOF_total ! it could be incorrect here
            else
               fct = 1.0d0
            endif
            aa_xx = dexp(-wd_t8(j)*chain_v_logs_xx(imol,inos+1)) + wd_t4(j)*chain_g_logs_xx(imol,inos)*aa_xx
            aa_yy = dexp(-wd_t8(j)*chain_v_logs_yy(imol,inos+1)) + wd_t4(j)*chain_g_logs_yy(imol,inos)*aa_yy
            aa_zz = dexp(-wd_t8(j)*chain_v_logs_zz(imol,inos+1)) + wd_t4(j)*chain_g_logs_zz(imol,inos)*aa_zz
            chain_g_logs_xx(imol,inos+1) = chain_v_logs_xx(imol,inos)*chain_v_logs_xx(imol,inos) *  fct - 1.0d0/tau2
            chain_g_logs_yy(imol,inos+1) = chain_v_logs_yy(imol,inos)*chain_v_logs_yy(imol,inos) *  fct - 1.0d0/tau2
            chain_g_logs_zz(imol,inos+1) = chain_v_logs_zz(imol,inos)*chain_v_logs_zz(imol,inos) *  fct - 1.0d0/tau2
          end do
          chain_v_logs_xx(imol,N_N_O_S) = chain_v_logs_xx(imol,N_N_O_S) + chain_g_logs_xx(imol,N_N_O_S)*wd_t4(j)
          chain_v_logs_yy(imol,N_N_O_S) = chain_v_logs_yy(imol,N_N_O_S) + chain_g_logs_yy(imol,N_N_O_S)*wd_t4(j)
          chain_v_logs_zz(imol,N_N_O_S) = chain_v_logs_zz(imol,N_N_O_S) + chain_g_logs_zz(imol,N_N_O_S)*wd_t4(j)
        end do
       enddo
      do iat = start_group(imol),end_group(imol)
      if(l_proceed(iat)) then
        vxx(iat) = vxx(iat) * csi_xx
        vyy(iat) = vyy(iat) * csi_yy
        vzz(iat) = vzz(iat) * csi_zz
      endif
      enddo
  endif !  dof_mol > 0
  enddo !i  = 1, Nmols

   end subroutine nhcint_masive_chains_mol_xyz

   subroutine nhcint_masive_chains_mol

 use kinetics, only : get_instant_temperature, add_kinetic_pressure
      use kinetics, only : get_instant_temperature, add_kinetic_pressure
      use thermostat_Lucretius_data
      use ALL_atoms_data, only : Natoms, vxx,vyy,vzz, all_atoms_mass,atom_dof
      use ALL_mols_data, only :  Nmols, mol_dof,start_group,end_group
      use integrate_data, only : time_step
      use ensamble_data, only : thermo_coupling, T_eval, temperature
      use DOF_data
      use physical_constants, only : Red_Boltzmann_constant

      integer i,j,inos,iat,kk
      real(8) csi,aa, tau2,fct  ,s_dof, s_2kin ,di0,idi0,px,py,pz,px_idi0,py_idi0,pz_idi0,imass
      real(8) T_imposed , Tx(Nmols), vxx_thermo(Natoms),vyy_thermo(Natoms),vzz_thermo(Natoms)
                                
      T_imposed = temperature

      call get_instant_temperature(T_eval)
      call add_kinetic_pressure

      tau2 = (thermo_coupling/time_step)**2
      ff = 1.0d0 / Red_Boltzmann_constant

      do imol = 1, Nmols
      csi_xx = 1.0d0
      if (mol_dof(imol)>0) then
      s_2kin = 0.0d0 
      do iat = start_group(imol),end_group(imol)      
      if(l_proceed(iat)) &
        s_2kin = s_2kin + all_atoms_mass(iat)*(vxx(iat)**2 + vyy(iat)**2 + vzz(iat)**2 )  
      enddo
      Tx(imol) =  s_2kin / mol_dof(imol)  * ff
      
      chain_g_logs_xx(imol,1) = (Tx(imol)/T_imposed - 1.0d0)/tau2
      do i = 1,nhc_step1
        do j = 1,nhc_step2
          chain_v_logs_xx(imol,N_N_O_S) = chain_v_logs_xx(imol,N_N_O_S)+chain_g_logs_xx(imol,N_N_O_S)*wd_t4(j)
          do inos = 1, N_N_O_S-1
            aa = dexp(-wd_t8(j)*chain_v_logs_xx(imol,N_N_O_S+1-inos))
            chain_v_logs_xx(imol,N_N_O_S-inos) = chain_v_logs_xx(imol,N_N_O_S-inos)*aa*aa + wd_t4(j)*chain_g_logs_xx(imol,N_N_O_S-inos)*aa
          end do
          aa_xx = dexp(-wd_t2(j)*chain_v_logs_xx(imol,1));
          csi_xx = csi_xx*aa_xx
          chain_g_logs_xx(imol,1) = (csi_xx*csi_xx*Tx(imol)/T_imposed - 1.0d0)/tau2
          chain_logs_xx(imol,1:N_N_O_S) = chain_logs_xx(imol,1:N_N_O_S) + chain_v_logs_xx(imol,1:N_N_O_S)* wd_t2(j)
          do inos = 1, N_N_O_S-1
            if (inos==1) then
               fct = mol_dof(imol) !DOF_total ! it could be incorrect here
            else
               fct = 1.0d0
            endif
            aa_xx = dexp(-wd_t8(j)*chain_v_logs_xx(imol,inos+1)) + wd_t4(j)*chain_g_logs_xx(imol,inos)*aa_xx
            chain_g_logs_xx(imol,inos+1) = chain_v_logs_xx(imol,inos)*chain_v_logs_xx(imol,inos) *  fct - 1.0d0/tau2
          end do
          chain_v_logs_xx(imol,N_N_O_S) = chain_v_logs_xx(imol,N_N_O_S) + chain_g_logs_xx(imol,N_N_O_S)*wd_t4(j)
        end do
       enddo

  do iat = start_group(imol), end_group(imol)
  if (l_proceed(iat)) then
      vxx_thermo(iat) = vxx(iat) * (csi_xx-1.0d0)
      vyy_thermo(iat) = vyy(iat) * (csi_xx-1.0d0)
      vzz_thermo(iat) = vzz(iat) * (csi_xx-1.0d0)
  endif
  enddo
  endif !  dof_mol > 0
  enddo !i  = 1, Nmols

!  px=0.0d0;py=0.0d0;pz=0.0d0;di0=0.0d0
!  do i = 1, Natoms
!  if (l_proceed(i))then
!    di0=di0+1.0d0
!    px = px+vxx_thermo(i)*all_atoms_mass(i)
!    py = py+vyy_thermo(i)*all_atoms_mass(i)
!    pz = pz+vzz_thermo(i)*all_atoms_mass(i)
!  endif
!  enddo
!  idi0=1.0d0/di0; px_idi0 = px*idi0; py_idi0=py*idi0; pz_idi0=pz*idi0;
!  do i = 1, Natoms
!  if (l_proceed(i))then
!    imass=1.0d0/all_atoms_mass(i)
!    vxx_thermo(i) = vxx_thermo(i) - px_idi0*imass
!    vyy_thermo(i) = vyy_thermo(i) - py_idi0*imass
!    vzz_thermo(i) = vzz_thermo(i) - pz_idi0*imass
!  endif
!  enddo

  do i = 1, Natoms
  if (l_proceed(i))then
    vxx(i) = vxx(i) + vxx_thermo(i)
    vyy(i) = vyy(i) + vyy_thermo(i)
    vzz(i) = vzz(i) + vzz_thermo(i)
  endif
  enddo

end subroutine nhcint_masive_chains_mol



   subroutine nptint
     use kinetics, only : get_kinetic_energy_stress,add_kinetic_pressure
      use thermostat_Lucretius_data
      use ALL_atoms_data, only : Natoms, vxx,vyy,vzz
      use integrate_data, only : time_step
      use ensamble_data, only : thermo_coupling, T_eval, temperature, barostat_coupling,&
          pressure_xx,pressure_yy,pressure_zz,pressure_ALL
      use DOF_data
      use energies_data
      use stresses_data
      use boundaries, only : cel_properties

      integer inos,iat,kk,i,j
      real(8) aa,cons,ak(3),boxmv2,trvlogv,t(3),const_press,ekin2,onf,baro2,tau2,d3,akin2(3)
 
      call get_kinetic_energy_stress(T_eval)
      call add_kinetic_pressure
      call cel_properties(.true.)
      
      tau2 = (thermo_coupling/time_step)**2
      baro2 = (barostat_coupling(1)/time_step)**2
      d3 = (DOF_total+3.0d0)/DOF_total
      onf = 1.0d0/DOF_total
      boxmv2 = dot_product(v_logv(1:3),v_logv(1:3)) *  baro2 * time_step
      ekin2 = 2.0d0*en_kin     
      const_press = (DOF_total+3.0d0) * T_imposed*Red_Boltzman_constant / baro2

      g_logs(1)=(T_eval/T_imposed + d3*( boxmv2-1.0d0 ) )/tau2

      g_logv(1) = ((onf*ekin2)+stress_kin(1)+(pressure(1)-pressure_xx)*Volume) * const_press
      g_logv(2) = ((onf*ekin2)+stress_kin(2)+(pressure(2)-pressure_yy)*Volume) * (const_press/3.0d0)
      g_logv(3) = ((onf*ekin2)+stress_kin(3)+(pressure(3)-pressure_zz)*Volume) * (const_press/3.0d0) 
      


      do i = 1,nhc_step1
        do j = 1,nhc_step2
          v_logs(N_N_O_S) = v_logs(N_N_O_S)+g_logs(N_N_O_S)*wd_t4(j)
          do inos = 1, N_N_O_S-1
            aa = dexp(-wd_t8(j)*v_logs(N_N_O_S+1-inos))
            v_logs(N_N_O_S-inos) = v_logs(N_N_O_S-inos)*aa*aa + wd_t4(j)*g_logs(N_N_O_S-inos)*aa
          end do
          aa = dexp(-wd_t8(j)*v_logs(1))
          v_logv(1:3)  = v_logv(1:3)*aa*aa + wd_t4(j)*g_logv(1:3)*aa
          trvlogv = onf*sum(v_logv(1:3))
          
          t(1:3) = dexp(-wd_t2(j)*(v_logs(1)+trvlogv+v_logv(1:3))) ; 
          akin2(1:3) = akin2(1:3)*t(1:3)
          vxx(:) = vxx(:) * t(1)
          vyy(:) = vyy(:) * t(2)
          vzz(:) = vzz(:) * t(3)

          
          ekin2 = sum(akin2(1:3))
          g_logv(1) = ((onf*ekin2)+stress_kin(1)+(pressure(1)-pressure_xx)*Volume) * const_press
          g_logv(2) = ((onf*ekin2)+stress_kin(2)+(pressure(2)-pressure_yy)*Volume) * const_press
          g_logv(3) = ((onf*ekin2)+stress_kin(3)+(pressure(3)-pressure_zz)*Volume) * const_press 
          
          x_logs(1:N_N_O_S) = x_logs(1:N_N_O_S) + v_logs(1:N_N_O_S)* wd_t2(j)
          aa = dexp(-wd_t8(j)*v_logs(1))
          v_logv(1:3)  = v_logv(1:3)*aa*aa + wd_t4(j)*g_logv(1:3)*aa
          boxmv2 = dot_product(v_logv(1:3),v_logv(1:3)) *  (baro2 * time_step)
          g_logs(1)=(T_eval/T_imposed + d3*( boxmv2-1.0d0 ) )/tau2
          do inos = 1, N_N_O_S-1
            if (inos==1) then
               fct = DOF_total ! it could be incorrect here
            else
               fct = 1.0d0
            endif
            aa =dexp(-wd_t8(j)*v_logs(inos+1))
            v_logs(inos) = v_logs(inos)*aa*aa+ wd_t4(j)*g_logs(inos)*aa
            g_logs(inos+1) = v_logs(inos)*v_logs(inos) *  fct - 1/tau2 
          end do
          v_logs(N_N_O_S) = v_logs(N_N_O_S) + g_logs(N_N_O_S)*wd_t4(j)
        end do
      end do

 end subroutine nptint
      
  
end module Lucretius_integrator_module

