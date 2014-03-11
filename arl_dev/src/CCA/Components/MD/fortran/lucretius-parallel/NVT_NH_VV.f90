
 subroutine NVT_NH_VV(stage)
! do the numerical integration of eq of motion in NVE using velocity verlet algorithm
! use 2 stages
! stage 1:
!    v(t+0.5*dt) = v(t) + 0.5*dt*f(t)/m
!    r(t+dt)    = r(t) + dt*v(t+0.5*dt) 
! stage 2:
!    v(t+dt)   = v(t+0.5*dt) + 0.5*dt*f(t+dt)
! between stage 1 and stage 2 the force evaluation is called
! this subroutine integrate as following:
! 1   call NVE_VV(stage=1) (forces from previous step are used)
! 2   call forces evaluation
! 3   call NVE_VV(stage=2)

 use ALL_atoms_data, only : xxx,yyy, zzz, vxx,vyy,vzz,fxx,fyy,fzz,Natoms, all_atoms_mass, is_dummy
 use integrate_data, only : time_step
 use kinetics
 use DOF_data
 use ensamble_data, only : temperature,thermo_coupling, T_eval
 use non_bonded_lists_data, only : l_update_VERLET_LIST

 implicit none
 integer, intent(IN) :: stage ! 1 or 2 
 real(8), allocatable :: xx1(:),yy1(:),zz1(:),vx(:),vy(:),vz(:)
 real(8) tmp, half_time_step, invstep, csi, T_imposed
 integer i,j,k
 logical l_proceed(Natoms)

 allocate(vx(Natoms),vy(Natoms),vz(Natoms))
 do i = 1, Natoms
 vx(i)  = vxx(i) ; vy(i)  = vyy(i) ; vz(i)  = vzz(i)
 enddo 

 do i = 1, Natoms
   l_proceed(i) = .not.(is_dummy(i).or.l_WALL_CTRL(i).or.all_atoms_mass(i)<1.0d-8)
 enddo


! SHAKE INITIALIZATION HERE
! END SHAKE INITIALIZATION 

 SELECT CASE (stage)

 CASE(1)
   allocate(xx1(Natoms),yy1(Natoms),zz1(Natoms)) ; 
   xx1(1:Natoms) = xxx(1:Natoms) ; yy1(1:Natoms) = yyy(1:Natoms) ; zz1(1:Natoms) = zzz(1:Natoms)
   half_time_step = time_step * 0.5d0
   do i = 1, Natoms
   if(l_proceed(i)) then
      tmp = half_time_step / all_atoms_mass(i)
      vxx(i) = vx(i)+tmp*fxx(i)   !  v(t+0.5*dt) ; the forces are from previous time-step
      vyy(i) = vy(i)+tmp*fyy(i)
      vzz(i) = vz(i)+tmp*fzz(i)
      xxx(i) = xx1(i) + time_step * vxx(i)   ! r(t+dt)
      yyy(i) = yy1(i) + time_step * vyy(i)
      zzz(i) = zz1(i) + time_step * vzz(i)
   else
     vxx(i)=0.0d0
     vyy(i)=0.0d0
     vzz(i)=0.0d0
   endif
   enddo
   call is_list_2BE_updated(l_update_VERLET_LIST)
   if (N_constrains>0)then
     call shake_vv_1 ! double check
     call is_list_2BE_updated(l_update_VERLET_LIST)
   endif

! APPLY SHAKE HERE
! END SHAKE 
   invstep = 1.0d0/time_step
   do i = 1, Natoms  ! correct velocities
   if(l_proceed(i)) then
     vxx(i) = (xxx(i)-xx1(i))*invstep
     vyy(i) = (yyy(i)-yy1(i))*invstep
     vzz(i) = (zzz(i)-zz1(i))*invstep
   else
     vxx(i)=0.0d0
     vyy(i)=0.0d0
     vzz(i)=0.0d0
   endif
   enddo
   deallocate(xx1,yy1,zz1)
  CASE(2) 
!   EXTERNAL FIELD CORRECTIONS HERE 
    half_time_step = time_step * 0.5d0
    do i = 1, Natoms
     if(l_proceed(i)) then
      tmp = half_time_step / all_atoms_mass(i)
      vxx(i) = vx(i)+tmp*fxx(i)   ! v(t+dt) ;  the forces are re-evaluated for t+dt
      vyy(i) = vy(i)+tmp*fyy(i)
      vzz(i) = vz(i)+tmp*fzz(i)
     else
      vxx(i)=0.0d0
      vyy(i)=0.0d0
      vzz(i)=0.0d0
     endif
    enddo

! APPLY SHAKE HERE
! END SHAKE

    call get_instant_temperature(T_eval)
    T_imposed = temperature
    csi=sqrt(1.d0+time_step/thermo_coupling*(T_imposed/T_eval-1.d0))
    vxx(1:Natoms)=vxx(1:Natoms)*csi
    vyy(1:Natoms)=vyy(1:Natoms)*csi
    vzz(1:Natoms)=vzz(1:Natoms)*csi
    call get_kinetic_energy_stress
 
      
 CASE default
   print*, 'ERRROR in NVE_VV; the variable stage can only be 1 or 2 and cannot be its actual value',stage
   STOP
 end SELECT

 end subroutine NVT_B_VV
