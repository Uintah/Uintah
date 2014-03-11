module intramol_constrains
 implicit none

 private :: get_bonds
 public :: shake_vv_1
 public :: shake_vv_2
 public :: shake_initializations_vv
 public :: shake_ll

 contains

      subroutine shake_vv_1
      use shake_data
      use ALL_atoms_data, only : Natoms, xxx,yyy,zzz, contrains_per_atom, all_atoms_massinv, vxx,vyy,vzz
      use connectivity_type_data, only : prm_constrain_types, constrain_types
      use connectivity_ALL_data, only : Nconstrains, list_constrains
      use stresses_data, only : stress_shake
      use profiles_data, only : atom_profile, l_need_2nd_profile, l_need_1st_profile 
      use integrate_data   
      use boundaries, only : periodic_images
      implicit none

      real(8) lsxx,lsxy,lsxz,& 
              lsyy,lsyz,lszz, sxx, syy, szz,sxy,sxz,syz 
      real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:),tx(:),ty(:),tz(:)
      real(8) XXDD,XXDD_sq,Ai,Aj,x,y,z,x1,y1,z1,r2,r_dot_rin,ff,ffi,ffj,coef1,CCi,CCj,di,dj
      real(8) error,max_error,term
      integer i,j,k,iat,jat,itype,iter,imax,istart,iend
      logical l_converge

      
      if (Nconstrains < 1) RETURN
!Nconstrains=2

      istart = 1 
      iend   = 6 !6 !Nconstrains
      allocate(dx(Nconstrains),dy(Nconstrains),dz(Nconstrains),dr_sq(Nconstrains))
      allocate(tx(Natoms),ty(Natoms),tz(Natoms))

      lsxx=0.0d0 ; lsxy=0.0d0 ; lsxz = 0.0d0 ; 
      lsyy=0.0d0 ; lsyz=0.0d0 ; lszz = 0.0d0 ; 

      iter=0
      l_converge=.false.
! start main shake cycle

      do while(.not.l_converge.and.iter<MX_SHAKE_ITERATIONS)
        iter=iter+1
        call get_bonds(dx(1:Nconstrains),dy(1:Nconstrains),dz(1:Nconstrains))
        dr_sq(1:Nconstrains) = dx(1:Nconstrains)*dx(1:Nconstrains)+&
                               dy(1:Nconstrains)*dy(1:Nconstrains)+&
                               dz(1:Nconstrains)*dz(1:Nconstrains)
       
        error=0.d0             
        max_error = -1.0d0; imax = -1
        do i= istart,iend !1,Nconstrains
          itype=list_constrains(0,i)
          XXDD= prm_constrain_types(1,itype)          
          term = dabs(dr_sq(i)-XXDD*XXDD)

          error = error + term
print*,i,XXDD,dsqrt(dr_sq(i))
! if (mod(i,100)==0)read(*,*)
          if ( term > max_error) then 
            max_error = term
            imax = i
          endif
        enddo        
print*, 'error=', error, dsqrt(error)/dble(Nconstrains), 'max =',max_error,'imax=',imax
print*, 'compare_bonds=',dsqrt(dr_sq(imax)),prm_constrain_types(1,list_constrains(0,imax))
read(*,*)
!print*, dsqrt(dr_sq(imax-1)),prm_constrain_types(1,list_constrains(0,imax-1)) ,&
! dsqrt(dr_sq(imax-1))-prm_constrain_types(1,list_constrains(0,imax-1))
!print*, dsqrt(dr_sq(imax+1)),prm_constrain_types(1,list_constrains(0,imax+1)) ,&
! dsqrt(dr_sq(imax+1))-prm_constrain_types(1,list_constrains(0,imax+1))


        l_converge = (error < SHAKE_TOLERANCE)
!print*, iter,'error TOP l_con=',error,SHAKE_TOLERANCE,l_converge
        if (.not.l_converge )then
          tx(:)=0.0d0; ty(:)=0.0d0; tz(:)=0.0d0; 
          do i=istart,iend !1,Nconstrains
            itype=list_constrains(0,i)
            XXDD= prm_constrain_types(1,itype)          
            iat=list_constrains(1,i)
            jat=list_constrains(2,i)
                      
            XXDD_sq = XXDD*XXDD
            CCi= time_step*all_atoms_massinv(iat)
            CCj=-time_step*all_atoms_massinv(jat)
            coef1 = -time_step*(CCi-CCj)
           
            x=dx(i)        ;   y=dy(i)       ;    z=dz(i) 
            r2 = dr_sq(i)
            x1 = dx_in(i)  ;  y1 = dy_in(i)  ;    z1 = dz_in(i)           
            r_dot_rin = x*x1+y*y1+z*z1
            ff=(XXDD_sq-r2)/(coef1*r_dot_rin)
            
            sxx = ff*x1*x1 ; sxy = ff*x1*y1 ; sxz = ff*x1*z1
                             syy = ff*y1*y1 ; syz = ff*y1*z1
                                              szz = ff*z1*z1 
                       
            lsxx=lsxx-sxx
            lsxy=lsxy-sxy
            lsxz=lsxz-sxz
            lsyy=lsyy-syy
            lsyz=lsyz-syz
            lszz=lszz-szz
                        
            ffi=-0.5d0*ff*CCi  
            tx(iat)= x1*ffi!tx(iat)+x1*ffi
            ty(iat)= y1*ffi!ty(iat)+y1*ffi
            tz(iat)= z1*ffi!tz(iat)+z1*ffi
            
            ffj=-0.5d0*ff*CCj  
            tx(jat) = x1*ffj!tx(jat)+x1*ffj
            ty(jat) = y1*ffj!ty(jat)+y1*ffj
            tz(jat) = z1*ffj!tz(jat)+z1*ffj
            
!          enddo !do i=1,Nconstrains

!          do i= istart,iend !1,Nconstrains            
            iat=list_constrains(1,i)
            jat=list_constrains(2,i)
            Ai=1.0d0/dble(contrains_per_atom(iat))
            Aj=1.0d0/dble(contrains_per_atom(jat))
Ai=1.0d0;Aj=1.0d0
            di = time_step * Ai ; dj = time_step * Aj
            xxx(iat)=xxx(iat)+di*tx(iat)
            yyy(iat)=yyy(iat)+di*ty(iat)
            zzz(iat)=zzz(iat)+di*tz(iat)
            xxx(jat)=xxx(jat)+dj*tx(jat)
            yyy(jat)=yyy(jat)+dj*ty(jat)
            zzz(jat)=zzz(jat)+dj*tz(jat)
            vxx(iat)=vxx(iat)+Ai*tx(iat)
            vzz(iat)=vzz(iat)+Ai*ty(iat)
            vyy(iat)=vyy(iat)+Ai*tz(iat)
            vxx(jat)=vxx(jat)+Aj*tx(jat)
            vyy(jat)=vyy(jat)+Aj*ty(jat)
            vzz(jat)=vzz(jat)+Aj*tz(jat)
          enddo
          
        endif
        
      enddo

      if(.not.l_converge) then
write(6,*) 'ERROR in shake_vv_1 : NO CONVERGENCE!!!!!'
STOP
      endif 
      
      stress_shake(1)=lsxx
      stress_shake(2)=lsyy
      stress_shake(3)=lszz
      stress_shake(4)=(lsxx+lsyy+lszz)/3.0d0
      stress_shake(5)=lsxy
      stress_shake(6)=lsxz
      stress_shake(7)=lsyz
      stress_shake(8)=lsxy
      stress_shake(9)=lsxz
      stress_shake(10) = lsyz

      deallocate(dx,dy,dz,dr_sq)
      deallocate(tx,ty,tz)
print*,'succesfully finished shake 1 in iter=',iter
STOP
      end subroutine shake_vv_1

      subroutine shake_vv_2
      use shake_data
      use ALL_atoms_data, only : Natoms, xxx,yyy,zzz, contrains_per_atom, all_atoms_massinv, vxx,vyy,vzz
      use connectivity_type_data, only : prm_constrain_types, constrain_types
      use connectivity_ALL_data, only : Nconstrains, list_constrains
      use stresses_data, only : stress_shake
      use profiles_data, only : atom_profile, l_need_2nd_profile, l_need_1st_profile
      use integrate_data
      use boundaries, only : periodic_images

      implicit none

      real(8) local_stress_xx,local_stress_xy,local_stress_xz,&
              local_stress_yy,local_stress_yz,local_stress_zz, sxx, syy, szz,sxy,sxz,syz
      real(8), allocatable :: dx(:),dy(:),dz(:),tx(:),ty(:),tz(:)
      real(8) Ai,Aj,x,y,z,x1,y1,z1,r2,dv_dot_rin,ff,ffi,ffj,coef1,CCi,CCj,di,dj
      real(8) error, local_tolerance, r2_1
      integer i,j,k,iat,jat,itype,iter
      logical l_converge

      if (Nconstrains < 1) RETURN
      allocate(dx(Nconstrains),dy(Nconstrains),dz(Nconstrains))
      allocate(tx(Natoms),ty(Natoms),tz(Natoms))

 
      local_tolerance=SHAKE_TOLERANCE/time_step
      iter=0
      l_converge=.false.
      do while(.not.l_converge.and.iter<MX_SHAKE_ITERATIONS)
      iter = iter + 1
      tx(:) = 0.0d0; ty(:) = 0.0d0 ; tz(:) = 0.0d0
      error=0.d0
      do i=1,Nconstrains
          iat=list_constrains(1,i)
          jat=list_constrains(2,i)
          x1 = dx_in(i)  ;  y1 = dy_in(i)  ;    z1 = dz_in(i)
          r2_1 = dr_sq_in(i)
          dv_dot_rin = x1*(vxx(iat)-vxx(jat))+y1*(vyy(iat)-vyy(jat))+z1*(vzz(iat)-vzz(jat))
          CCi= time_step*all_atoms_massinv(iat)
          CCj=-time_step*all_atoms_massinv(jat)
          coef1 = (CCi-CCj)*r2_1
          ff = dv_dot_rin / coef1 
          error = error + dabs(ff)          
          ffi=-0.5d0*ff*CCi
          tx(iat)=tx(iat)+x1*ffi
          ty(iat)=ty(iat)+y1*ffi
          tz(iat)=tz(iat)+z1*ffi
          ffj=-0.5d0*ff*CCj
          tx(jat)=tx(jat)+x1*ffj
          ty(jat)=ty(jat)+y1*ffj
          tz(jat)=tz(jat)+z1*ffj
       enddo
       l_converge = error < local_tolerance
       do i = 1, Nconstrains
          iat=list_constrains(1,i)
          jat=list_constrains(2,i)
          Ai=1.0d0/dble(contrains_per_atom(iat))
          Aj=1.0d0/dble(contrains_per_atom(jat))
          vxx(iat)=vxx(iat)+Ai*tx(iat)
          vyy(iat)=vyy(iat)+Ai*ty(iat)
          vzz(iat)=vzz(iat)+Ai*tz(iat)
          vxx(jat)=vxx(jat)+Aj*tx(jat)
          vyy(jat)=vyy(jat)+Aj*ty(jat)
          vzz(jat)=vzz(jat)+Aj*tz(jat)
        enddo
      enddo ! while

      if(.not.l_converge) then
write(6,*) 'ERROR in shake_vv_2 : NO CONVERGENCE!!!!!'
STOP
      endif

      deallocate(dx,dy,dz)
      deallocate(tx,ty,tz)

      end  subroutine shake_vv_2


      subroutine get_bonds(dx,dy,dz)
      use ALL_atoms_data, only : xxx,yyy,zzz
      use connectivity_ALL_data, only : Nconstrains, list_constrains
      use boundaries, only : periodic_images
      implicit none
      real(8), intent(out) :: dx(Nconstrains),dy(Nconstrains),dz(Nconstrains)
      integer i,iat,jat

       do i=1,Nconstrains
          iat=list_constrains(1,i)
          jat=list_constrains(2,i)
          dx(i)=xxx(iat)-xxx(jat)
          dy(i)=yyy(iat)-yyy(jat)
          dz(i)=zzz(iat)-zzz(jat)
       enddo
       call periodic_images(dx(1:Nconstrains),dy(1:Nconstrains),dz(1:Nconstrains))
       end subroutine get_bonds

       subroutine shake_initializations_vv
        use shake_data
        use connectivity_ALL_data, only : Nconstrains
        use boundaries, only : periodic_images
        implicit none
        call get_bonds(dx_in,dy_in,dz_in)
        dr_sq_in(:) = dx_in(:)*dx_in(:)+dy_in(:)*dy_in(:)+dz_in(:)*dz_in(:)
       end subroutine shake_initializations_vv

subroutine shake_ll(stress_update,xxx1,yyy1,zzz1,xxx_shaken,yyy_shaken,zzz_shaken)
! shake with multiple time step integration
      use ALL_atoms_data, only : xxx,yyy,zzz, Natoms,all_atoms_massinv,all_atoms_mass
      use connectivity_ALL_data, only : Nconstrains, list_constrains
      use connectivity_type_data, only : prm_constrain_types
      use boundaries, only : periodic_images
      use shake_data, only :MX_SHAKE_ITERATIONS,SHAKE_TOLERANCE, shake_iterations
      use rolling_averages_data, only : RA_shake_iter
      use generic_statistics_module, only : statistics_5_eval
      use integrate_data, only : integration_step

      implicit none
      logical,intent(IN):: stress_update
      real(8), intent(INOUT) :: xxx1(:),yyy1(:),zzz1(:)
      real(8), intent(INOUT) :: xxx_shaken(:),yyy_shaken(:),zzz_shaken(:)
      integer i,j,k,itype,iat,jat,istart,iend,iter,imax
      real(8) maxi
      real(8) inv_mass_red, di,dj,g,XXDD_sq,XXDD,ERROR
      logical l_converge
      real(8),allocatable:: dx1(:),dy1(:),dz1(:),dr_sq(:)
      real(8),allocatable:: dx0(:),dy0(:),dz0(:),dr_sq0(:)
      real(8),allocatable:: diff(:),cross(:)
      real(8),allocatable:: xxx1_saved(:),yyy1_saved(:),zzz1_saved(:)
      real(8) time_step_short 

 
      call local_initializations
      call get_bonds_0
       !! include numaromatic_1  
      istart = 1
      iend = Nconstrains

      iter=0
      do while((.not.l_converge).and.iter<MX_SHAKE_ITERATIONS)
        iter=iter+1
!        call get_bonds_1
!        cross(:) = dx1(:)*dx0(:)+dy1(:)*dy0(:)+dz1(:)*dz0(:)
        imax=0
        do i=istart,iend !1,Nconstrains
            itype=list_constrains(0,i)
            XXDD= prm_constrain_types(1,itype)          
            iat=list_constrains(1,i) ; jat=list_constrains(2,i)
            XXDD_sq = XXDD*XXDD
            dx1(i)=xxx1(iat)-xxx1(jat)
            dy1(i)=yyy1(iat)-yyy1(jat)
            dz1(i)=zzz1(iat)-zzz1(jat)
            call periodic_images(dx1(i:i),dy1(i:i),dz1(i:i))  !DAMIT
            dr_sq(i:i) = dx1(i:i)*dx1(i:i)+dy1(i:i)*dy1(i:i)+dz1(i:i)*dz1(i:i) !DAMIT
            cross(i) = dx1(i)*dx0(i) + dy1(i)*dy0(i) + dz1(i)*dz0(i)
            diff(i) = XXDD_sq - dr_sq(i)
            inv_mass_red = (all_atoms_massinv(iat) + all_atoms_massinv(jat))*2.0d0
            g = diff(i)/(cross(i)*inv_mass_red)
            di =  g*all_atoms_massinv(iat)  ;   dj = -g*all_atoms_massinv(jat)
            xxx1(iat) = xxx1(iat) + di * dx0(i)
            yyy1(iat) = yyy1(iat) + di * dy0(i)
            zzz1(iat) = zzz1(iat) + di * dz0(i)
            xxx1(jat) = xxx1(jat) + dj * dx0(i)
            yyy1(jat) = yyy1(jat) + dj * dy0(i)
            zzz1(jat) = zzz1(jat) + dj * dz0(i)
         enddo
         ERROR = dsqrt(maxval(dabs(diff(istart:iend))))
         l_converge = (ERROR < SHAKE_TOLERANCE)
      enddo ! while main iterative cycle
            
        
      xxx_shaken = xxx1 ; yyy_shaken = yyy1 ; zzz_shaken = zzz1 
     
!print*,'stress_update=',stress_update 
      if (stress_update) then
        call local_stress_update   
      endif 

      shake_iterations = dble(iter)
      call statistics_5_eval(RA_shake_iter, shake_iterations)

      
      if (.not.l_converge) then
        do i = istart,iend
         print*, 'i diff(i)=',i,diff(i)
        enddo
        print*, 'maxval|diff|=',maxval(dabs(diff(istart:iend)))
        print*, 'dsqrt(maxval|diff|)=',ERROR
        print*, 'ERROR in shake_ll: shake NOT convergent in ',MX_SHAKE_ITERATIONS, ' at integration step ', integration_step
        STOP
      endif   
       call clean_up_and_finalize
  
CONTAINS

     subroutine local_initializations
     use integrate_data, only : time_step
     use thermostat_Lucretius_data, only : Multi_Big 
      allocate(dx1(Nconstrains),dy1(Nconstrains),dz1(Nconstrains),dr_sq(Nconstrains))
      allocate(dx0(Nconstrains),dy0(Nconstrains),dz0(Nconstrains),dr_sq0(Nconstrains))
      allocate(diff(Nconstrains),cross(Nconstrains))
      allocate(xxx1_saved(Natoms),yyy1_saved(Natoms),zzz1_saved(Natoms))
      l_converge=.false.
      xxx1_saved=xxx1; yyy1_saved=yyy1 ; zzz1_saved=zzz1
      time_step_short = time_step/dble(Multi_Big)
    end subroutine local_initializations
    
    subroutine clean_up_and_finalize  
      deallocate(dx1,dy1,dz1,dr_sq)
      deallocate(dx0,dy0,dz0,dr_sq0)
      deallocate(diff,cross)
      deallocate(xxx1_saved,yyy1_saved,zzz1_saved)
    end subroutine clean_up_and_finalize  

    subroutine get_bonds_0
      use boundaries, only : periodic_images
      implicit none

      integer i,iat,jat,N
      N=Nconstrains

       do i=1,N
          iat=list_constrains(1,i)
          jat=list_constrains(2,i)
          dx0(i)=xxx(iat)-xxx(jat)
          dy0(i)=yyy(iat)-yyy(jat)
          dz0(i)=zzz(iat)-zzz(jat)
       enddo
       call periodic_images(dx0(1:N),dy0(1:N),dz0(1:N))
       dr_sq0(1:N) = dx0(1:N)*dx0(1:N)+dy0(1:N)*dy0(1:N)+dz0(1:N)*dz0(1:N)
    end subroutine get_bonds_0
       
    subroutine get_bonds_1
      use boundaries, only : periodic_images
      implicit none

      integer i,iat,jat,N
      N=Nconstrains
       do i=1,N
          iat=list_constrains(1,i)
          jat=list_constrains(2,i)
          dx1(i)=xxx1(iat)-xxx1(jat)
          dy1(i)=yyy1(iat)-yyy1(jat)
          dz1(i)=zzz1(iat)-zzz1(jat)
       enddo
       call periodic_images(dx1(1:N),dy1(1:N),dz1(1:N))
       dr_sq(1:N) = dx1(1:N)*dx1(1:N)+dy1(1:N)*dy1(1:N)+dz1(1:N)*dz1(1:N)
    end subroutine get_bonds_1        
       
    subroutine local_stress_update
       use boundaries, only : periodic_images
       use stresses_data, only : stress, stress_shake
       use ALL_atoms_data, only :  map_from_intramol_constrain_to_atom, &
                                   any_intramol_constrain_per_atom
       use sizes_data, only : Natoms_with_intramol_constrains
       integer i,j,k,i1
       real(8) fx,fy,fz,x,y,z,ff,idt
       real(8) sxx,sxy,sxz,syx,syy,syz,szx,szy,szz
       real(8) lsxx,lsxy,lsxz,lsyx,lsyy,lsyz,lszx,lszy,lszz
       real(8), allocatable :: xxx2(:),yyy2(:),zzz2(:)
       idt=1.0d0/(time_step_short*time_step_short)
        lsxx=0.0d0 ; lsxy=0.0d0 ; lsxz = 0.0d0 ; 
        lsyx=0.0d0 ; lsyy=0.0d0 ; lsyz = 0.0d0 ; 
        lszx=0.0d0 ; lszy=0.0d0 ; lszz = 0.0d0 ; 
        allocate(xxx2(Natoms_with_intramol_constrains),yyy2(Natoms_with_intramol_constrains),&
        zzz2(Natoms_with_intramol_constrains))
        i1=0
        do i = 1, Natoms
        if (any_intramol_constrain_per_atom(i)) then
          i1 = i1 + 1
          xxx2(i1) = xxx1(i); yyy2(i1) = yyy1(i); zzz2(i1) = zzz1(i)
        endif
        enddo
!        call periodic_images(xxx2(1:i1),yyy2(1:i1),zzz2(1:i1))        
        do i1 = 1,Natoms_with_intramol_constrains
          i = map_from_intramol_constrain_to_atom(i1)
          ff = all_atoms_mass(i) * idt
!print*,i1,'ff=',ff/418.4,all_atoms_mass(i),time_step_short
!print*,'dist=',xxx1(i),xxx1_saved(i)
!print*,'dist=',yyy1(i),yyy1_saved(i)
!print*,'dist=',zzz1(i),zzz1_saved(i)
!read(*,*)
!stop
          x = xxx2(i1); y = yyy2(i1); z = zzz2(i1)
          fx = ff * (xxx1(i)-xxx1_saved(i))  ; 
          fy = ff * (yyy1(i)-yyy1_saved(i))  ;
          fz = ff * (zzz1(i)-zzz1_saved(i))
          sxx = fx * x ; sxy = fx * y ; sxz = fx * z
          syx = fy * x ; syy = fy * y ; syz = fy * z
          szx = fz * x ; szy = fz * y ; szz = fz * z
!print*,fx/418.4,fy/418.4,fz/418.4
!print*,sxx/418.4,sxy/418.4,sxz/418.4
!print*,syx/418.4,syy/418.4,syz/418.4
!print*,szx/418.4,szy/418.4,szz/418.4
!read(*,*)
          lsxx=lsxx+sxx ; lsxy=lsxy+sxy ; lsxz = lsxz+sxz ; 
          lsyx=lsyx+syx ; lsyy=lsyy+syy ; lsyz = lsyz+syz ; 
          lszx=lszx+szx ; lszy=lszy+szy ; lszz = lszz+szz ; 
        enddo
       deallocate(xxx2,yyy2,zzz2)
       stress_shake(1) = lsxx
       stress_shake(2) = lsyy
       stress_shake(3) = lszz
       stress_shake(4) = (lsxx+lsyy+lszz)/3.0d0
       stress_shake(5) = lsxy
       stress_shake(6) = lsxz
       stress_shake(7) = lsyz
       stress_shake(8) = lsyx
       stress_shake(9) = lszx
       stress_shake(10)= lszy
       stress(:) = stress(:) + stress_shake(:)
!print*,'stress_shake=',stress_shake/418.4d0
!read(*,*)
     end subroutine local_stress_update


end subroutine shake_ll

end module intramol_constrains
