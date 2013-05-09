module write_stuff
implicit none
private :: get_file_names_4_writting
private :: write_instant_boxed_xyz
private :: write_12_profiles
private :: write_scalar_statistics
private :: write_average_xyz_file
private :: write_instant_atomic_energy
!private :: remove_file
public :: writting_driver
public :: write_config
public :: quick_preview_statistics
public :: write_history
 contains 

 subroutine writting_driver
 use file_names_data

   call get_file_names_4_writting
!print*, trim(name_out_config_file)
!print*, trim(name_xyz_all_atoms_file)
   call write_config
   call write_instant_boxed_xyz
   call write_instant_atomic_energy
   call write_12_profiles
   call write_scalar_statistics
   call write_average_xyz_file

 end subroutine writting_driver

! subroutine remove_file(nf) ! to call a c function I have to make it private object; not sure why
! character(*), intent(IN) :: nf
! integer i
! i = rm_file(trim(nf))
! print*,i,trim(nf)
! end subroutine remove_file 

 subroutine quick_preview_statistics
 use integrate_data, only : integration_step, time_step
 use rolling_averages_data, only : RA_temperature,RA_pressure,RA_energy,RA_sfc
 use quick_preview_stats_data, only : quick_preview_stats
 use file_names_data, only : nf_quick_preview_stat,path_out,continue_job
 use sim_cel_data, only : Volume, Area_xy
 use physical_constants, only : unit_pressure,Red_Vacuum_EL_permitivity_4_Pi,Volt_to_internal_field
 use field_constrain_data, only : N_atoms_field_constrained
 implicit none
 logical :: first_time = .true.
 integer i
 integer, save :: NNN
 character(100),save :: format_
 real(8) T,pz,p,Ep,Et,q1sfc,q2sfc,fi1,fi2
 real(8), save :: fp , fe, fq, ff
 type tag_type
   real(8)::val
   real(8)::counts
 end type tag_type
 type(tag_type), save :: tagT,tagpz,tagp,tagEp,tagEt,tagq1sfc,tagq2sfc,tagfi1,tagfi2

 if (mod(integration_step,quick_preview_stats%how_often)==0) then
 if (first_time) then
   first_time=.false.
   nf_quick_preview_stat = trim(path_out)//'preview'//'_'//trim(continue_job%field1%ch)
   fp = (unit_pressure /Volume/ 1.0d5 / 1.0d3)
   fe = (0.01d0 /1000.0d0 ) ! In MJ/mol
   fq = dsqrt( Red_Vacuum_EL_permitivity_4_Pi) / Area_xy * 100.0d0 ! in e-/100A^2
   ff = 1.0d0 / Volt_to_internal_field
   do i = 1, 100; format_(i:i) = ' '; enddo
   format_ = '(I10,1X,F9.4,1X,F8.3,1X,2(F9.5,1X),2(F15.5,1X),4(F10.6,1X))'
   NNN = len(trim(format_))
!   i = rm_file(trim(nf_quick_preview_stat))
!   call remove_file(trim(nf_quick_preview_stat))
   open(unit=22,file=trim(nf_quick_preview_stat),status='replace',recl=200)
   write(22,*) '    step  |time(ps)| T (K) |pz(1000atm)|pxy(1000atm)|    Ep(MJ/mol) | Et(MJ/mol) | q1(sfc) |q2(sfc) e-/100A^2| fi1   |   fi2 (V) '
   close(22)
   call default_all_tags
 endif

  open(unit=22,file=trim(nf_quick_preview_stat), access='append',recl=200)
  T =  (RA_temperature%val-tagT%val)  / (RA_temperature%counts-tagT%counts )
  pz = (RA_pressure(3)%val-tagpz%val) / (RA_pressure(3)%counts-tagpz%counts) * fp
  p  = (RA_pressure(4)%val-tagp%val)  / (RA_pressure(4)%counts-tagp%counts ) * fp
  Ep = (RA_energy(3)%val-tagEp%val)   / (RA_energy(3)%counts-tagEp%counts  ) * fe
  Et = (RA_energy(5)%val-tagEt%val)   / (RA_energy(5)%counts-tagEt%counts  ) * fe
  if (N_atoms_field_constrained > 0 ) then
   q1sfc = (RA_sfc(1)%val - tagq1sfc%val) / (RA_sfc(1)%counts-tagq1sfc%counts) *fq
   q2sfc = (RA_sfc(2)%val - tagq2sfc%val) / (RA_sfc(2)%counts-tagq2sfc%counts) *fq
  else
   q1sfc=0.0d0; q2sfc=0.0d0;
  endif
  fi1 = (RA_sfc(3)%val - tagfi1%val) / (RA_sfc(3)%counts - tagfi1%counts) * ff
  fi2 = (RA_sfc(4)%val - tagfi2%val) / (RA_sfc(4)%counts - tagfi2%counts) * ff
  write(22,format_(1:NNN))integration_step, integration_step*time_step,T,pz,0.5d0*(p*3.0d0-pz),Ep,Et,q1sfc,q2sfc,fi1,fi2
  close(22)
  call save_all_tags
 endif

 contains
   subroutine default_tag(tag)
     implicit none
     type(tag_type), intent(INOUT) :: tag
     tag%val=0.0d0
     tag%counts=0.0d0
   end subroutine default_tag
   subroutine default_all_tags
     call default_tag(tagT)
     call default_tag(tagpz)
     call default_tag(tagp)
     call default_tag(tagEp)
     call default_tag(tagEt)
     call default_tag(tagq1sfc)
     call default_tag(tagq2sfc)
     call default_tag(tagfi1) 
     call default_tag(tagfi2)
   end subroutine default_all_tags
   subroutine save_all_tags
      tagT%val     = RA_temperature%val
      tagT%counts  = RA_temperature%counts
      tagpz%val    = RA_pressure(3)%val
      tagpz%counts = RA_pressure(3)%counts
      tagp%val     = RA_pressure(4)%val
      tagp%counts  = RA_pressure(4)%counts
      tagEp%val    = RA_energy(3)%val
      tagEp%counts = RA_energy(3)%counts
      tagEt%val    = RA_energy(5)%val
      tagEt%counts = RA_energy(5)%counts
      tagq1sfc%counts=RA_sfc(1)%counts
      tagq2sfc%counts=RA_sfc(2)%counts
      tagq1sfc%val=RA_sfc(1)%val
      tagq2sfc%val=RA_sfc(2)%val
      tagfi1%counts = RA_sfc(3)%counts
      tagfi2%counts = RA_sfc(4)%counts
      tagfi1%val = RA_sfc(3)%val
      tagfi2%val = RA_sfc(4)%val
   end subroutine save_all_tags
 end subroutine quick_preview_statistics

 subroutine get_file_names_4_writting
  use chars, only : char_intN_ch_NOBLANK, char_intN_ch
  use file_names_data
  implicit none
 
!  continue_job%field1%i = continue_job%field1%i + 1

  continue_job%field2%i = continue_job%field2%i + 1
  continue_job%keep_indexing = continue_job%keep_indexing + 1
  call char_intN_ch(4,continue_job%field2%i,continue_job%field2%ch)

 name_out_config_file = trim(path_out)//'config'//'_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 name_xyz_all_atoms_file = trim(A_path)//'xyz'//'_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)//'.xyz'
 name_xyz_file_just_mol=trim(M_path)//'MOL_xyz'//'_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)//'.xyz'

 nf_atom_density     =  trim(z_A_path)//'A_dens_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_atom_energy = trim(A_path)//'A_energy_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_mol_energy  = trim(M_path)//'M_energy_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_mol_density      =  trim(z_M_path)//'M_dens_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_atom_temperature =  trim(z_A_path)//'A_Temp_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_mol_charges      =  trim(z_M_path)//'M_charge_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_atom_charges     =  trim(z_A_path)//'A_charge_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_atom_dipoles     =  trim(z_A_path)//'A_dipole_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_atom_pot         =  trim(z_A_path)//'A_pot_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_atom_field       =  trim(z_A_path)//'A_field_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_atom_pot_Q       =  trim(z_A_path)//'A_potQ_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_mol_pot          =  trim(z_M_path)//'M_pot_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_mol_pot_Q        =  trim(z_M_path)//'M_potQ_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_mol_force        =  trim(z_M_path)//'M_force_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_mol_stress       =  trim(z_M_path)//'M_stress_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_poisson_field    = trim(z_M_path)//'poisson_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_A_poisson_field  = trim(z_A_path)//'A_poisson_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_mol_OP1          = trim(z_M_path)//'M_OP1_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch) 

 nf_zp_MOL_rsmd      = trim(z_M_path)//'rsmd_zp_m_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_mol_MOL_rsmd     = trim(M_path)//'rsmd_MOL_m_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)
 nf_xyz_av_file      = trim(A_path)//'xyz_AVG_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)//'.xyz'  
 nf_atom_fi          = trim(A_path)//'A_fi_'//trim(continue_job%field1%ch)//'_'//trim(continue_job%field2%ch)

 print*, trim(name_out_config_file)
  
 end subroutine get_file_names_4_writting

 subroutine write_more_energies 
 use integrate_data
 use energies_data
 use file_names_data, only : nf_more_energies
 use ensamble_data, only : T_eval,Temperature_trans_Calc,Temperature_rot_Calc,temperature
 use thetering_data, only : thetering
 implicit none
 integer N
 logical, save :: first_time=.true.
 if (first_time) then
   first_time=.false.
   call put_header
 endif

! usefull for energies conservation  
 if (l_do_QN_CTRL) then 
   N=500
 else
   N=1500
 endif
   open(unit=66,file=trim(nf_more_energies),access='append',recl=1000)
   if (l_do_QN_CTRL) then 
     T_eval = 0.5d0*(Temperature_trans_Calc+Temperature_rot_Calc)
     En_kin = En_kin_rotation + En_kin_translation
   endif
 if (l_do_QN_CTRL) then
 if (thetering%N > 0) then
   write(66,'(I7,1X,4(F14.8,1X),F10.5,1X,F10.5,1X,3(F14.8,1X))')&
         integration_step, En_kin/100.0d0/1000.0d0,(en_pot)/100.0d0/1000.0d0,&
         ( En_pot + en_kin)/100.0d0/1000.0d0, en_water_surface_extra/100.0d0/1000.0d0,T_eval, temperature,&
         En_Q/1.0d5, En_vdw/1.0d5,en_thetering/1.0d5
 else
    write(66,'(I7,1X,4(F14.8,1X),F10.5,1X,F10.5,1X,2(F14.8,1X))')&
         integration_step, En_kin/100.0d0/1000.0d0,(en_pot)/100.0d0/1000.0d0,&
         ( En_pot + en_kin)/100.0d0/1000.0d0, en_water_surface_extra/100.0d0/1000.0d0,T_eval, temperature,&
         En_Q/1.0d5, En_vdw/1.0d5
 endif
 else 
 if (thetering%N > 0) then
   write(66,'(I7,1X,4(F13.7,1X),F10.5,1X,3(F13.7,1X), 2(F10.5,1X),3(F13.7,1X))')&
         integration_step, En_kin/100.0d0/1000.0d0,(en_pot)/100.0d0/1000.0d0,&
         ( En_pot + en_kin)/100.0d0/1000.0d0, en_water_surface_extra/100.0d0/1000.0d0,T_eval,temperature, &
          en_bond/1.0d5, en_angle/1.0d5, en_dih/1.0d5, en_deform/1.0d5, &
          En_Q/1.0d5, En_vdw/1.0d5,en_thetering/1.0d5
 else
   write(66,'(I7,1X,4(F13.7,1X),F10.5,1X,3(F13.7,1X), 2(F10.5,1X),2(F13.7,1X))')&
         integration_step, En_kin/100.0d0/1000.0d0,(en_pot)/100.0d0/1000.0d0,&
         ( En_pot + en_kin)/100.0d0/1000.0d0, en_water_surface_extra/100.0d0/1000.0d0,T_eval,temperature, &
          en_bond/1.0d5, en_angle/1.0d5, en_dih/1.0d5, en_deform/1.0d5, &
          En_Q/1.0d5, En_vdw/1.0d5
 endif
 endif
   close(66)

 contains 
 subroutine put_header
 open(unit=66,file=trim(nf_more_energies),recl=1000)
   if (l_do_QN_CTRL) then
   if (thetering%N > 0) then
   write(66,*) &
   ' step En_kin   en_pot    En_tot    en_surf_multibody    T_eval  T_imposed  Q vdw  thetering (MJ/mol) '
   else
   write(66,*) &
   ' step En_kin   en_pot    En_tot    en_surf_multibody    T_eval  T_imposed  Q vdw  (MJ/mol) '
   endif
 else
 if (thetering%N > 0) then
   write(66,*)&
    ' step En_kin en_pot En_tot en_surf_multibody T_eval T_imposed en_bond/ en_angle en_dih en_deform Q vdw thetering (MJ/mol)'
 else 
    write(66,*)&
    ' step En_kin en_pot En_tot en_surf_multibody T_eval T_imposed en_bond/ en_angle en_dih en_deform Q vdw (MJ/mol)'
 endif
 endif
 close(66)
 end subroutine put_header

 end subroutine write_more_energies

 subroutine write_config
 use file_names_data, only : continue_job
 use sim_cel_data, only : sim_cel, i_boundary_CTRL
 use ensamble_data
 use ALL_atoms_data
 use atom_type_data, only : atom_type_name
 use field_constrain_data
 use file_names_data, only : name_out_config_file, FILE_continuing_jobs_indexing, path_out
 use sizes_data, only : Nmols
 use ALL_rigid_mols_data, only : mol_MOM,mol_ANG,qn
 use integrate_data
 use ALL_mols_data, only : mol_xyz
 use CTRLs_data
 use thetering_data

 implicit none
 integer i
 character(500) instruction 
 instruction = ""
  open(unit=33,file=trim(trim(path_out)//trim(FILE_continuing_jobs_indexing)))
  write(33,*) continue_job%field1%i, continue_job%field2%i, continue_job%keep_indexing
  close(33)

! Do here the conting of continue job
  open(unit=33,file=trim(name_out_config_file), recl = 500)

  write(33,*) 'SIM_BOX'
  write(33,*) sim_cel(1:3)
  write(33,*) sim_cel(4:6)
  write(33,*) sim_cel(7:9)     
  write(33,*) i_boundary_CTRL
  write(33,*)
  write(33,*) 'INDEX_CONTINUE_JOB'
  write(33,*) continue_job%field1%i, continue_job%field2%i, continue_job%keep_indexing
  write(33,*) 
  write(33,*) 'MOLECULES_AND_ATOMS'
  write(33,*) Natoms, Nmols
  write(33,*)
  write(33,*) 'ENSAMBLE'
  write(33,*) i_type_ensamble_CTRL
  write(33,*)
  write(33,*) 'THERMOSTAT'
  write(33,*) i_type_thermostat_CTRL
  select case(i_type_thermostat_CTRL)
  case(1)   ! BERDENSEN
  case(2)   ! NOSE HOOVER
     write(33,*)  thermo_position, thermo_velocity, thermo_force
  case(3)   ! NOSE HOOVER CHAIN
  end select

  write(33,*) 'BAROSTAT'
  write(33,*) i_type_barostat_CTRL
  select case(i_type_barostat_CTRL)
  case(1)   ! BERDENSEN anisotropic
  case(2)   ! BERDENSEN ISOTROPIC
  case(3)   ! NOSE HOOVER STYLE
     write(33,*) baro_position
     write(33,*) baro_velocity
     write(33,*) baro_force
  end select
  write(33,*)
  write(33,*) 'XYZ'
  do i = 1, Natoms
     write(33,*) xxx(i),yyy(i),zzz(i) , '   ',trim(atom_type_name(i_type_atom(i))),'   ',i
  enddo
  write(33,*)

 if (l_do_QN_CTRL) then
   write(33,*) 'MOL_XYZ'
   do i = 1, Nmols  
     write(33,*) mol_xyz(i,:)
   enddo
   write(33,*) 
   write(33,*) 'QN'
   do i = 1, Nmols
     write(33,*) qn(i,:)
   enddo
   write(33,*)
   write(33,*) 'MOL_MOM'
   do i = 1, Nmols
     write(33,*) mol_MOM(i,:)
   enddo
   write(33,*)
   write(33,*) 'MOL_ANG'
   do i = 1, Nmols
     write(33,*) MOL_ANG(i,:)
   enddo
   write(33,*)
 endif



  write(33,*)
if (.not.l_do_QN_CTRL) then
  write(33,*) 'VXYZ'
  do i = 1, Natoms
     write(33,*) vxx(i),vyy(i),vzz(i)
  enddo
  write(33,*)
endif

if (l_ANY_Q_pol_CTRL.or.l_ANY_S_FIELD_CONS_CTRL) then 
  write(33,*) 'CHARGES'
  do i = 1, Natoms
     write(33,*) all_charges(i)
  enddo
endif
  write(33,*)
if (l_ANY_DIPOLE_POL_CTRL) then
  write(33,*) 'DIPOLES'
  do i = 1, Natoms
     write(33,*) all_dipoles_xx(i),all_dipoles_yy(i),all_dipoles_zz(i),is_dipole_polarizable(i)
  enddo
  write(33,*)
endif

 write(33,*) 'BASE_XYZ '
   do i = 1, Natoms
     write(33,*) base_dx(i),base_dy(i),base_dz(i)
  enddo

 if (thetering%N>0) then
    write(33,*) 'THETERING'
    write(33,*) thetering%N
    do i = 1, thetering%N
      write(33,*) thetering%to_atom(i),thetering%x0(i),thetering%y0(i),thetering%z0(i),&
                  thetering%kx(i),thetering%kz(i),thetering%ky(i),'  ',i,'   ',trim(atom_type_name(i_type_atom(i)))
    enddo
 endif 

 close(33)

instruction = "cp "//trim(name_out_config_file)//" "//trim(path_out)//"config.final"
call SYSTEM(instruction)

 end subroutine write_config

 subroutine write_instant_boxed_xyz
   use ALL_atoms_data  , only : Natoms, xx,yy,zz,i_type_atom
   use boundaries, only : periodic_images
   use atom_type_data, only : atom_type_name
   use sim_cel_data
   use file_names_data, only : name_xyz_all_atoms_file,path_out
   implicit none
   character(4) ch4
   integer i,j,k
   character(350) instruction
   instruction = "";
   open(unit=22,file=trim(name_xyz_all_atoms_file))
   write(22,*) Natoms
   write(22,'(3(F9.5,1X))')sim_cel(1),sim_cel(5),sim_cel(9)
   do i = 1, Natoms
     ch4 = atom_type_name(i_type_atom(i))(1:4)
     write(22,'(A4,1X,3(F9.4,1X))') ch4, xx(i),yy(i),zz(i)
   enddo
   close(22)
 
 instruction = "cp "//trim(name_xyz_all_atoms_file)//" "//trim(path_out)//"xyz.final"
 call SYSTEM(instruction)
  
 end subroutine write_instant_boxed_xyz

 subroutine write_12_profiles
 use profiles_data
 use ALL_atoms_data
 use ALL_mols_data 
 use mol_type_data, only : N_type_molecules
 use atom_type_data, only : N_type_atoms
 use physical_constants
 use math_constants, only : Pi4
 use file_names_data
 use sim_cel_data
 use collect_data
 use general_statistics_data
 use poisson_1D, only : poisson_field, poisson_field_eval, poisson_A_field
 use CTRLs_data, only : l_DIP_CTRL
 use rsmd_data, only : rmsd_qn_med,rmsd_qn_med_2,rmsd_xyz_med,rmsd_xyz_med_2,zp_rmsd_xyz_med_2,zp_rmsd_xyz_med
 use integrate_data, only : l_do_QN_CTRL
 use sizes_data, only :  N_type_atoms_for_statistics, N_type_mols_for_statistics
 use rsmd_data, only : zp_translate_cryterion

 implicit none
 ! atom density
 integer i,j,k,NRECL
 real(8) v,vct4(4),vct10(10),vct9(9),v1,v2,v3,v4,d,fp,ff,temp,ct,is1
 real(8) , allocatable :: z_ax(:),z_ax_s(:),VV(:),VV2(:),poisson_q(:),x_ax(:),y_ax(:),VVx(:),VVy(:),VV2x(:),VV2y(:)
 character(500) instruction 
 instruction =""

 allocate(Z_ax(N_BINS_ZZ),z_ax_s(N_BINS_ZZs),VV(N_BINS_ZZ),VV2(N_BINS_ZZ),poisson_q(N_BINS_ZZ))
allocate(VVx(N_BINS_XX),VV2x(N_BINS_XX),VVy(N_BINS_YY),VV2y(N_BINS_YY))
 allocate(x_ax(N_BINS_xx),y_ax(N_BINS_YY));
 do i = 1, N_BINS_ZZ
   Z_ax(i) = dble(i)/dble(N_BINS_ZZ) * sim_cel(9)
 enddo 
 do i = 1, N_BINS_ZZs
    Z_ax_s(i) = dble(i)/dble(N_BINS_ZZs) * sim_cel(9)
 enddo

 do i = 1, N_BINS_XX
   X_ax(i) = dble(i)/dble(N_BINS_XX) * sim_cel(1)
 enddo
 do i = 1, N_BINS_YY
   Y_ax(i) = dble(i)/dble(N_BINS_YY) * sim_cel(5)
 enddo


NRECL = 10000

if (l_1st_profile_CTRL) then
 if (N_type_atoms_for_statistics>0)then
 open(unit = 333,file=trim(nf_atom_density),recl=NRECL)
 open(unit = 334,file=trim(nf_atom_density)//'.X',recl=NRECL)
 open(unit = 335,file=trim(nf_atom_density)//'.Y',recl=NRECL)
 do i = 1, N_BINS_ZZ
  write(333,'(F11.6)',advance='no') z_ax(i)
  do j = 1, N_type_atoms_for_statistics
    if (counter_ATOMS_global(i,j) == 0) then
      V = 0.0d0
    else
      V = zp1_atom(i)%density(j) / di_collections * RESCALE_DENSITY
    endif

    write(333,'(1X,E17.10)',advance='no') V
  enddo
  write(333,*)
 enddo
 close(333)
 instruction = "cp "//trim(nf_atom_density)//" "//trim(path_out)//"A_dens.final"
 call SYSTEM(instruction)


 do i = 1, N_BINS_XX
 write(334,'(F11.6)',advance='no') x_ax(i)
  do j = 1, N_type_atoms_for_statistics
    if (counter_ATOMS_global_x(i,j) == 0) then
      V = 0.0d0
    else
      V = zp1_atom_x(i)%density(j) / di_collections * RESCALE_DENSITY
    endif
    write(334,'(1X,E17.10)',advance='no') V
  enddo
  write(334,*)
 enddo
 close(334)

 do i = 1, N_BINS_YY
 write(335,'(F11.6)',advance='no') y_ax(i)
  do j = 1, N_type_atoms_for_statistics
    if (counter_ATOMS_global_y(i,j) == 0) then
      V = 0.0d0
    else
      V = zp1_atom_y(i)%density(j) / di_collections * RESCALE_DENSITY
    endif
    write(335,'(1X,E17.10)',advance='no') V
  enddo
  write(335,*)
 enddo
 close(335)


 endif



 
 if (N_type_mols_for_statistics>0)then
 open(unit = 333,file=trim(nf_mol_OP1),recl=NRECL)
 do i = 1, N_BINS_ZZ
  write(333,'(F11.6)',advance='no') z_ax(i)
  do j = 1, N_type_mols_for_statistics
  do k = 1, 6
    if (counter_MOLS_global(i,j) == 0) then
      V = 0.0d0
    else
      d = 1.0d0/dble(counter_MOLS_global(i,j))
      V = zp1_mol(i)%OP(j,k) * d
    endif
    write(333,'(1X,F16.8)',advance='no') V
  enddo
  enddo
  write(333,*)
 enddo
 close(333)
 endif

 ! atom temperature
 if (.not.l_do_QN_CTRL)then
 open(unit = 333, file = trim(nf_atom_temperature),recl=NRECL)
 do i = 1, N_BINS_ZZ
   write(333,'(F11.6)',advance='no') z_ax(i)
     if (zp1_atom(i)%DOF == 0.0d0.or.is_dummy(i) ) then
       v = 0.0d0
     else
       v = 2.0d0* zp1_atom(i)%kin / (Red_Boltzmann_constant * zp1_atom(i)%DOF) ! in K
     endif
     write(333,'(1X,F11.4)',advance='no') v
   write(333,*)
  enddo
 close(333)

instruction = "cp "//trim(nf_atom_temperature)//" "//trim(path_out)//"temperature_A.final"
call SYSTEM(instruction)


 endif
 ! mol  charges
  is1 = 1.0d0/(sim_cel(1)*sim_cel(5)) * 100.0d0 ! in 100A^2 (nm^2)
  do i = 1, N_BINS_ZZ
    VV(i) = zp1_mol(i)%p_charge(1)+zp1_mol(i)%g_charge(1)
  enddo 
  open(unit = 333, file = trim(nf_mol_charges),recl=NRECL)
  ff = dsqrt(Red_Vacuum_EL_permitivity_4_Pi) / di_collections  * is1  !(to make in in e-/100A^2)
  ct = electron_charge/Vacuum_EL_permitivity /unit_length/Pi4 / 100.0d0
  do i = 1, N_BINS_ZZ
   write(333,'(F11.6)',advance='no') z_ax(i)
   do j = 1, 4
       v =  zp1_mol(i)%p_charge(j) * ff ! in e-/A^2
     write(333,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  zp1_mol(i)%G_charge(j) * ff ! in e-/A^2
     write(333,'(1X,F10.5)',advance='no') v
   enddo ! j
   write(333,'(1X,F10.5)',advance='no') sum(VV(1:i)) * ff
   write(333,'(1X,F10.5)',advance='no') sum(VV(N_BINS_ZZ-i+1:N_BINS_ZZ)) * ff
   write(333,*)
  enddo
 close(333)


 open(unit = 333, file = trim(nf_atom_charges),recl=NRECL)
 open(unit = 334, file = trim(nf_atom_charges)//'.X',recl=NRECL)
 open(unit = 335, file = trim(nf_atom_charges)//'.Y',recl=NRECL)
   do i = 1, N_BINS_ZZ
    VV(i) = zp1_atom(i)%p_charge(1)+zp1_atom(i)%g_charge(1)
  enddo
  do i = 1, N_BINS_ZZ
    VV2(i) = sum(VV(1:i))
  enddo

  do i = 1, N_BINS_ZZ
   write(333,'(F11.6)',advance='no') z_ax(i)
   do j = 1, 4
       v =  zp1_atom(i)%p_charge(j) * ff  ! in e-/A^2
     write(333,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  zp1_atom(i)%G_charge(j) * ff  ! in e-/A^2
     write(333,'(1X,F10.5)',advance='no') v
   enddo ! j
   v = (zp1_atom(i)%p_charge(1)+zp1_atom(i)%G_charge(1) ) * ff
   write(333,'(1X,F10.5)',advance='no') v
   write(333,'(1X,F10.5)',advance='no') sum(VV(1:i)) * ff
   write(333,'(1X,F10.5)',advance='no') sum(VV(N_BINS_ZZ-i+1:N_BINS_ZZ)) * ff
   write(333,'(1X,F10.5)',advance='no') -sum(VV2(1:i)) * (ff * ct * (z_ax(2)-z_ax(1))) *Pi4
   poisson_q(i) =  -sum(VV2(1:i)) * (ff * ct * (z_ax(2)-z_ax(1))) *Pi4
   
   write(333,*)
  enddo
 close(333)
 instruction="";
 instruction = "cp "//trim(nf_atom_charges)//" "//trim(path_out)//"A_charge.final"
 call SYSTEM(instruction)



  do i = 1, N_BINS_XX
    VVx(i) = zp1_atom_x(i)%p_charge(1)+zp1_atom_x(i)%g_charge(1)
  enddo
  do i = 1, N_BINS_XX
    VV2x(i) = sum(VVx(1:i))
  enddo

  do i = 1, N_BINS_XX
   write(334,'(F11.6)',advance='no') x_ax(i)
   do j = 1, 4
       v =  zp1_atom_x(i)%p_charge(j) * ff  ! in e-/A^2
     write(334,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  zp1_atom_x(i)%G_charge(j) * ff  ! in e-/A^2
     write(334,'(1X,F10.5)',advance='no') v
   enddo ! j
   v = (zp1_atom_x(i)%p_charge(1)+zp1_atom_x(i)%G_charge(1) ) * ff
   write(334,'(1X,F10.5)',advance='no') v
   write(334,'(1X,F10.5)',advance='no') sum(VVx(1:i)) * ff
   write(334,'(1X,F10.5)',advance='no') sum(VVx(N_BINS_XX-i+1:N_BINS_XX)) * ff
   write(334,'(1X,F10.5)',advance='no') -sum(VV2x(1:i)) * (ff * ct * (x_ax(2)-x_ax(1))) *Pi4
!   poisson_q(i) =  -sum(VV2x(1:i)) * (ff * ct * (x_ax(2)-x_ax(1))) *Pi4

   write(334,*)
  enddo
 close(334)
 instruction="";
 instruction = "cp "//trim(nf_atom_charges)//".X"//" "//trim(path_out)//"A_charge.X.final"
 call SYSTEM(instruction)



!!!!YYYY
  do i = 1, N_BINS_YY
    VVy(i) = zp1_atom_y(i)%p_charge(1)+zp1_atom_y(i)%g_charge(1)
  enddo
  do i = 1, N_BINS_YY
    VV2y(i) = sum(VVy(1:i))
  enddo

  do i = 1, N_BINS_YY
   write(335,'(F11.6)',advance='no') y_ax(i)
   do j = 1, 4
       v =  zp1_atom_y(i)%p_charge(j) * ff  ! in e-/A^2
     write(335,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  zp1_atom_y(i)%G_charge(j) * ff  ! in e-/A^2
     write(335,'(1X,F10.5)',advance='no') v
   enddo ! j
   v = (zp1_atom_y(i)%p_charge(1)+zp1_atom_y(i)%G_charge(1) ) * ff
   write(335,'(1X,F10.5)',advance='no') v
   write(335,'(1X,F10.5)',advance='no') sum(VVy(1:i)) * ff
   write(335,'(1X,F10.5)',advance='no') sum(VVy(N_BINS_YY-i+1:N_BINS_YY)) * ff
   write(335,'(1X,F10.5)',advance='no') -sum(VV2y(1:i)) * (ff * ct * (y_ax(2)-y_ax(1))) *Pi4
!   poisson_q(i) =  -sum(VV2x(1:i)) * (ff * ct * (x_ax(2)-x_ax(1))) *Pi4
   write(335,*)
  enddo
 close(335)
 instruction ="";
 instruction = "cp "//trim(nf_atom_charges)//".Y"//" "//trim(path_out)//"A_charge.Y.final"
 call SYSTEM(instruction)

if (l_DIP_CTRL) then
  open(unit = 333, file = trim(nf_atom_dipoles),recl=NRECL)
  open(unit = 334, file = trim(nf_atom_dipoles)//'.X',recl=NRECL)
  open(unit = 335, file = trim(nf_atom_dipoles)//'.Y',recl=NRECL)

  do i = 1, N_BINS_ZZ
    VV(i) = zp1_atom(i)%p_dipole(3)+zp1_atom(i)%g_dipole(3)
  enddo
  do i = 1, N_BINS_ZZ
   write(333,'(F11.6)',advance='no') z_ax(i)
   do j = 1, 4
       v =  zp1_atom(i)%p_dipole(j) * ff ! in e- Amstrom
     write(333,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  zp1_atom(i)%G_dipole(j) * ff ! in e- Amstrom
     write(333,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  (zp1_atom(i)%p_dipole(j)+zp1_atom(i)%G_dipole(j)) * ff ! in e- Amstrom
     write(333,'(1X,F10.5)',advance='no') v
   enddo ! j
   write(333,'(1X,F10.5)',advance='no') v
   write(333,'(1X,F10.5)',advance='no') sum(VV(1:i)) * ff
   write(333,'(1X,F10.5)',advance='no') (-sum(VV(1:i))+sum(VV2(1:i)))*ff*ct*(z_ax(2)-z_ax(1))*Pi4 
   write(333,*)
  enddo
 close(333)
 instruction=""
 instruction = "cp "//trim(nf_atom_dipoles)//" "//trim(path_out)//"A_dipole.final"
 call SYSTEM(instruction)

  do i = 1, N_BINS_XX
    VVx(i) = zp1_atom_x(i)%p_dipole(3)+zp1_atom_x(i)%g_dipole(3)
  enddo
  do i = 1, N_BINS_XX
   write(334,'(F11.6)',advance='no') x_ax(i)
   do j = 1, 4
       v =  zp1_atom_x(i)%p_dipole(j) * ff ! in e- Amstrom
     write(334,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  zp1_atom_x(i)%G_dipole(j) * ff ! in e- Amstrom
     write(334,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  (zp1_atom_x(i)%p_dipole(j)+zp1_atom_x(i)%G_dipole(j)) * ff ! in e- Amstrom
     write(334,'(1X,F10.5)',advance='no') v
   enddo ! j
   write(334,'(1X,F10.5)',advance='no') v
   write(334,'(1X,F10.5)',advance='no') sum(VVx(1:i)) * ff
   write(334,'(1X,F10.5)',advance='no') (-sum(VVx(1:i))+sum(VV2x(1:i)))*ff*ct*(x_ax(2)-x_ax(1))*Pi4
   write(334,*)
  enddo
 close(334)

  do i = 1, N_BINS_YY
    VVy(i) = zp1_atom_y(i)%p_dipole(3)+zp1_atom_y(i)%g_dipole(3)
  enddo
  do i = 1, N_BINS_YY
   write(335,'(F11.6)',advance='no') y_ax(i)
   do j = 1, 4
       v =  zp1_atom_y(i)%p_dipole(j) * ff ! in e- Amstrom
     write(335,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  zp1_atom_y(i)%G_dipole(j) * ff ! in e- Amstrom
     write(335,'(1X,F10.5)',advance='no') v
   enddo ! j
   do j = 1, 4
       v =  (zp1_atom_y(i)%p_dipole(j)+zp1_atom_y(i)%G_dipole(j)) * ff ! in e- Amstrom
     write(335,'(1X,F10.5)',advance='no') v
   enddo ! j
   write(335,'(1X,F10.5)',advance='no') v
   write(335,'(1X,F10.5)',advance='no') sum(VVy(1:i)) * ff
   write(335,'(1X,F10.5)',advance='no') (-sum(VVy(1:i))+sum(VV2y(1:i)))*ff*ct*(y_ax(2)-y_ax(1))*Pi4
   write(335,*)
  enddo
 close(335)

endif


 if (N_type_mols_for_statistics>0)then
 open(unit = 333,file=trim(nf_mol_density),recl=NRECL)
 do i = 1, N_BINS_ZZ
  write(333,'(F11.6)',advance='no') z_ax(i)
  do j = 1, N_type_mols_for_statistics
    if (counter_MOLS_global(i,j) == 0) then
      V = 0.0d0
    else
      V = zp1_mol(i)%density(j) / di_collections
    endif
    write(333,'(1X,E17.10)',advance='no') V
  enddo
  write(333,*)
 enddo
 close(333)
 endif
 instruction=""
 instruction = "cp "//trim(nf_mol_density)//" "//trim(path_out)//"M_dens.final"
 call SYSTEM(instruction)

! do poisson potential 
 
 open(unit = 333,file=trim(nf_A_poisson_field),recl=NRECL)
 do i = 1, N_BINS_ZZ
   write(333,'(F11.6)',advance='no') z_ax(i)
   write(333,'(2X,F12.7)',advance='no') poisson_q(i)
   write(333,*)
 enddo
 close(333)

 if (N_type_mols_for_statistics>0)then 
    open(unit = 333,file=trim(nf_zp_MOL_rsmd),recl=NRECL)
    do i = 1, N_BINS_ZZs
      write(333,'(F11.6)',advance='no') z_ax_s(i)
      do j = 1, N_type_mols_for_statistics
         write(333,'(2X,F12.7)',advance='no') zp_translate_cryterion(i,j)
!print*,i,j,zp_translate_cryterion(i,j)
      enddo
     write(333,*)
    enddo
  close(333)
 endif 

 open(unit = 333,file=trim(nf_mol_MOL_rsmd),recl=NRECL)
    do i = 1, Nmols
         temp = dot_product(rmsd_xyz_med(i,1:3),rmsd_xyz_med(i,1:3))/di_collections_short
         v1 = (rmsd_xyz_med_2(i) - temp)/di_collections_short
         write(333,'(I7,1X,F15.7,1X)',advance='yes') i,v1
    enddo
  close(333)
 
endif  ! l_need_first_profile

if (l_2nd_profile_CTRL) then

  open(unit = 333,file=trim(nf_atom_fi),recl=150)
    do i = 1, Natoms
         temp = atom_profile(i)%fi / Volt_to_internal_field
         write(333,'(I7,1X,2(F15.7,1X))',advance='yes') i,temp,RA_fi(i)/ Volt_to_internal_field/RA_fi_counts
    enddo
  close(333)

 if (N_type_atoms_for_statistics>0)then
 open(unit = 333, file=trim(nf_atom_pot),recl=NRECL)
 open(unit = 334, file=trim(nf_atom_field),recl=NRECL)
 open(unit = 335, file=trim(nf_atom_pot_Q),recl=NRECL)
 do i = 1, N_BINS_ZZ
  write(333,'(F11.6)',advance='no') z_ax(i)
  write(334,'(F11.6)',advance='no') z_ax(i)
  write(335,'(F11.6)',advance='no') z_ax(i)
  do j = 1, N_type_atoms_for_statistics
    if (counter_ATOMS_global(i,j) == 0) then
      v1 = 0.0d0
      v2 = 0.0d0
      v3 = 0.0d0
    else
      d = 1.0d0/counter_ATOMS_global(i,j)
      v1 = zp2_atom(i)%pot(j)  * d / 100.0d0
      v2 = zp2_atom(i)%fi(j) * d / Volt_to_internal_field
      v3 = zp2_atom(i)%Qpot(j) * d / 100.0d0
    endif
      write(333,'(1X,F15.6)',advance='no') v1
      write(334,'(1X,E15.6)',advance='no') v2
      write(335,'(1X,F15.6)',advance='no') v3
   enddo
  write(333,*)
  write(334,*)
  write(335,*)
 enddo
 close(333)
 close(334)
 close(335)
 endif !

 if (N_type_mols_for_statistics>0)then
 open(unit = 333, file=trim(nf_mol_pot),recl=NRECL)
 open(unit = 3331, file=trim(nf_mol_pot_Q),recl=NRECL)
 open(unit = 334, file=trim(nf_mol_force),recl=NRECL)
! open(unit = 335, file=trim(nf_mol_stress),recl=NRECL*2) !SKIP THE STRESSES FOR NOW
! open(unit = 336,file=trim(nf_mol_superficial_tension),recl=NRECL)

fp = unit_pressure/1.0d5/1.0d3 ! kBarr
do i = 1, N_BINS_ZZ
  write(333,'(F11.6)',advance='no') z_ax(i)
 write(3331,'(F11.6)',advance='no') z_ax(i)
  write(334,'(F11.6)',advance='no') z_ax(i)
!  write(335,'(F8.6)',advance='no') dble(i)/dble(N_BINS_ZZ)
!  write(336,'(F8.6)',advance='no') dble(i)/dble(N_BINS_ZZ)
 do j = 1, N_type_mols_for_statistics
    if (counter_MOLS_global(i,j) == 0) then
      v1 = 0.0d0
      vct4 = 0.0d0
!      vct10 = 0.0d0
    else
      d = 1.0d0/counter_MOLS_global(i,j)
      v1 = zp2_mol(i)%pot(j)  * d / 100.0d0  ! kJ/mol
      vct4(1:4) = zp2_mol(i)%force(j,1:4) * d / 100.0d0 ! kJ/mol/A
!      vct10(1:10) = zp2_mol(i)%stress(j,1:10) / di_collections * fp
    endif
     write(333,'(1X,F15.6)',advance='no') v1
     write(3331,'(1X,F15.6)',advance='no') zp2_mol(i)%Qpot(j)  * d / 100.0d0
     write(334,'(4(1X,F15.6))',advance='no') vct4(1:4)
!     write(335,'(9(1X,F10.5))',advance='no') vct9(1:9)
 enddo
 write(333,*)
 write(3331,*)
 write(334,*)
! write(335,*)
enddo

close(333)
close(3331)
close(334)
!close(335)
endif ! N_type_mols_for_statistics>0

endif    ! l_need_2nd_profile


deallocate(z_ax,z_ax_s,VV,VV2,poisson_q)
deallocate(x_ax,y_ax)
deallocate(VV2x,VVx,VV2y,VVy)
end subroutine write_12_profiles


subroutine write_average_xyz_file
use boundaries, only : periodic_images
use integrate_data, only : l_do_QN_CTRL
use ALL_atoms_data, only : Natoms, i_type_atom
use ALL_mols_data, only : start_group,end_group, i_type_molecule, Nmols
use mol_type_data, only : N_mols_of_type, N_type_atoms_per_mol_type, &
                          N_type_molecules, mol_type_xyz0
use atom_type_data, only : atom_type_name
use rsmd_data, only : rmsd_qn_med, rmsd_xyz_med
use sim_cel_data, only : sim_cel
use profiles_data, only : l_1st_profile_CTRL, l_need_1st_profile
use collect_data, only : di_collections_short
use file_names_data, only : nf_xyz_av_file
implicit none
real(8),allocatable :: av_atom_xyz(:,:),save_ra_qn(:,:), av_mol_orient(:,:)
real(8) dr(3)
integer i,j,k,i1,i2,i_sum,iii

  if (l_need_1st_profile) then

  allocate(av_atom_xyz(Natoms,3))

  if (l_do_QN_CTRL) then
        allocate(save_ra_qn(Nmols,4),av_mol_orient(Nmols,10))
        save_ra_qn(:,:) = rmsd_qn_med(:,:)/di_collections_short

        av_mol_orient(:,1)=-save_ra_qn(:,1)**2+save_ra_qn(:,2)**2-save_ra_qn(:,3)**2+save_ra_qn(:,4)**2
        av_mol_orient(:,2)=-2.0d0*(save_ra_qn(:,1)*save_ra_qn(:,2)+save_ra_qn(:,3)*save_ra_qn(:,4))
        av_mol_orient(:,3)=2.0d0*(save_ra_qn(:,2)*save_ra_qn(:,3)-save_ra_qn(:,1)*save_ra_qn(:,4))
        av_mol_orient(:,4)=2.0d0*(save_ra_qn(:,3)*save_ra_qn(:,4)-save_ra_qn(:,1)*save_ra_qn(:,2))
        av_mol_orient(:,5)=save_ra_qn(:,1)**2-save_ra_qn(:,2)**2-save_ra_qn(:,3)**2+save_ra_qn(:,4)**2
        av_mol_orient(:,6)=-2.0d0*(save_ra_qn(:,1)*save_ra_qn(:,3)+save_ra_qn(:,2)*save_ra_qn(:,4))
        av_mol_orient(:,7)=2.0d0*(save_ra_qn(:,2)*save_ra_qn(:,3)+save_ra_qn(:,1)*save_ra_qn(:,4))
        av_mol_orient(:,8)=2.0d0*(save_ra_qn(:,2)*save_ra_qn(:,4)-save_ra_qn(:,1)*save_ra_qn(:,3))
        av_mol_orient(:,9)=-save_ra_qn(:,1)**2-save_ra_qn(:,2)**2+save_ra_qn(:,3)**2+save_ra_qn(:,4)**2
 i1=0
  i2=0
  i_sum=0
  do i=1,N_type_molecules
   do iii=1,N_mols_of_type(i)
      i1=i1+1
      do j=1,N_type_atoms_per_mol_type(i)
        i2=i2+1
        i_sum = i_type_atom(i2)
!!!!!!!!!!!        i_type_atom(i2)=i_Class_atom(i_sum)
        dr(:)=mol_type_xyz0(i,j,:)
!        print*, 'dr=',dr
        av_atom_xyz(i2,1)= &
               av_mol_orient(i1,1)*dr(1) + av_mol_orient(i1,2)*dr(2) + av_mol_orient(i1,3)*dr(3)
        av_atom_xyz(i2,2)= &
               av_mol_orient(i1,4)*dr(1) + av_mol_orient(i1,5)*dr(2) + av_mol_orient(i1,6)*dr(3)
        av_atom_xyz(i2,3)= &
               av_mol_orient(i1,7)*dr(1) + av_mol_orient(i1,8)*dr(2) + av_mol_orient(i1,9)*dr(3)

      enddo
   enddo
   enddo
  deallocate(save_ra_qn,av_mol_orient)
  endif ! l_do_QN_CTRL

   do i=1,Nmols
   dr(:) = rmsd_xyz_med(i,:)/di_collections_short
   do j = start_group(i),end_group(i)
      av_atom_xyz(j,:)= dr(:)
   enddo
   enddo

   call periodic_images(av_atom_xyz(1:Natoms,1),av_atom_xyz(1:Natoms,2),av_atom_xyz(1:Natoms,3))


   open(unit=101,file=trim(nf_xyz_av_file))
   write(101,*) Natoms
   write(101,'(3(F10.5,1X))') sim_cel(1),sim_cel(5),sim_cel(9)

     do i=1,Natoms
       write(101,'(A4, 3(F9.4,1X) ) ') atom_type_name(i_type_atom(i)), av_atom_xyz(i,1:3)
     enddo
   close(101)


deallocate(av_atom_xyz)

 endif ! l_need_1st_profile
end subroutine write_average_xyz_file

subroutine write_instant_atomic_energy
use file_names_data, only : nf_atom_energy, nf_mol_energy
use ALL_atoms_data, only : Natoms
use ALL_mols_data, only : Nmols, mol_potential
use profiles_data, only : atom_profile, l_need_2nd_profile
implicit none
integer i
if (l_need_2nd_profile) then
open(unit=33,file=trim(nf_atom_energy),recl=200)
do i = 1, Natoms
write(33,*) i,atom_profile(i)%pot/1.0d5,atom_profile(i)%Qpot/1.0d5,atom_profile(i)%kin/1.0d5
enddo
close(33)

open(unit=33,file=trim(nf_mol_energy),recl=200)
do i = 1, Nmols
write(33,*) i,mol_potential(i)/1.0d5
enddo
close(33)

endif


end subroutine write_instant_atomic_energy



subroutine write_scalar_statistics
use file_names_data, only : path_out, continue_job, name_out_file
use integrate_data, only : integration_step
use rolling_averages_data
use energies_data
use stresses_data
use physical_constants
use cg_buffer, only : cg_iterations
use ensamble_data
use ALL_atoms_data, only : xxx,yyy,zzz,xx,yy,zz,fxx,fyy,fzz,all_atoms_mass,vxx,vyy,vzz,&
                           all_p_charges,all_g_charges
use CTRLs_data
use timing_module, only : ending_date, get_time_diference
use sim_cel_data
use integrate_data, only : l_do_QN_CTRL
use ALL_rigid_mols_data, only : mol_MOM
use RA1_stresses_data
use sizes_data
use sys_preparation_data, only : sys_prep
use thetering_data, only: thetering

implicit none
integer i,j,k,NNN
real(8) vct(200),vmax(200),vmin(200),t(3)
real(8) fp,fq, buf1,buf2,buf3
real(8) time_diff
character(100) format_
open(unit=1234,file=trim(name_out_file),access='append',recl=200)

call get_time_diference(time_diff) ! will also set the ending_date
fp = 0.01d0 /1000.0d0  ! convert in kJ/mol
write(1234,*) '\--------------     INTEGRATION STEP =',integration_step
write(1234,'(A15,A2,I5,2(A3,I3),A2,A1,2(I3,A3),I3)') &
   'DATE AND TIME :','y:',ending_date%year,': m',ending_date%month,': d',ending_date%day,'||',&
   'h',ending_date%hour,': m',ending_date%min,': s',ending_date%sec
do i = 1,100 ; format_(i:i) = ' '; enddo
format_(1:len('(A8,5(F18.7,1X))')) = '(A8,5(F18.7,1X))'
NNN = len(trim(format_))
write(1234,*) ' TIME ELAPSED (hours) ',time_diff, '   processing speed (iter/sec) =',dble(integration_step)/(time_diff*3600.0d0)
write(1234,*) ' !!! ******** ENERGY :  vdw Q pot kin tot : instant/RA/disp2/min/max   units=(1000 kJ/mol=1 MJ/mol)'
write(1234,format_(1:NNN)) 'instant=',en_vdw*fp, en_Q*fp, en_pot*fp, en_kin*fp, en_tot*fp
if(thetering%N>0) write(1234,*) 'en_thetering=',en_thetering*fp
write(1234,format_(1:NNN)) 'averagd=',RA_energy(1:5)%val/RA_energy(1:5)%counts*fp
if(thetering%N>0) write(1234,*) 'RA_en_thetering=',RA_energy(17)%val/RA_energy(17)%counts*fp ! 17 = thetering energy
do i = 1,5; call disp2(RA_energy(i),vct(i)); enddo 
write(1234,format_(1:NNN)) 'disp2 = ',dsqrt( vct(1:5)  ) * fp
vmax(1:5) = RA_energy(1:5)%max * fp
vmin(1:5) = RA_energy(1:5)%min * fp
write(1234,format_(1:NNN)) 'min=    ',vmax(1:5)
write(1234,format_(1:NNN)) 'min=    ',vmin(1:5)
write(1234,format_(1:NNN)) 'max-min=',vmax(1:5)-vmin(1:5)
if (.not.l_do_QN_CTRL) then
 write(1234,*) 'Energy intramol : bond angle dihedral deform  total_intramol:'
 write(1234,format_(1:NNN)) 'instant=',RA_energy(6:10)%val/RA_energy(6:10)%counts*fp
endif
write(1234,*) 'En Q details: En_Qreal En_Q_complx En_Q_complx_k=0 -En_Q_intra_corr -En_self ='
write(1234,format_(1:NNN)) ' Aver:Qdetails=', RA_energy(11:15)%val/RA_energy(11:15)%counts*fp
write(1234,format_(1:NNN)) ' Inst:Qdetails=', en_Qreal*fp, En_Q_cmplx*fp, En_Q_k0_cmplx*fp , En_Q_intra_corr*fp, &
      (En_Q_Gausian_self+ew_self)*fp
if (system_force_CTRL/=0) then
write(1234,*) 'Verify that the En_Q is equal with virial Q (units of MJ/mol)'
t(1:3) = (stress_Qreal(1:3)+stress_Qcmplx(1:3)+stress_Qcmplx_k_eq_0(1:3)-stress_excluded(1:3))
buf1=sum(t(1:3))
write(1234,*) 'instant en vir en-vir=',en_Q*fp, buf1*fp,(en_Q-buf1)*fp
t(1:3) = (RA1_stress_Qreal(1:3)+RA1_stress_Qcmplx(1:3)+RA1_stress_Qcmplx_k_eq_0(1:3)-RA1_stress_excluded(1:3))/RA1_stress_counts
buf1=sum(t(1:3))
buf2=RA_energy(2)%val/RA_energy(2)%counts
write(1234,*) 'average en vir en-vir=',buf2*fp, buf1*fp,(buf2-buf1)*fp
endif
!----------------------------------------------------------
fp = unit_pressure /Volume/ 1.0d5 / 1.0d3 ! 10^3 kBarr
!fp = 1.0d0/418.4d0 / 1000.0d0  ! in Mcal / mol
!fp = 1.0d0/100.0d0/1000.0d0    ! in MJ/mol
do i = 1,100 ; format_(i:i) = ' '; enddo
format_(1:len('(A8,10(F8.3,1X))')) = '(A8,10(F8.3,1X))'; 
NNN = len(trim(format_))
write(1234,*) ' !!! ******* INTERNAL STRESS :  xx yy zz | tot | xy xz yz  : instant/RA/disp2/min/max   units=(1000 atm)'
write(1234,format_(1:NNN)) 'instant=',stress(1:10) * fp
write(1234,format_(1:NNN)) 'averagd=',RA_stress(1:10)%val/RA_stress(1:10)%counts * fp
do i = 1,10 ; call disp2(RA_stress(i),vct(i)); enddo
write(1234,format_(1:NNN)) 'disp2 = ',dsqrt(vct(1:10) ) * fp
vmax(1:10) = RA_stress(1:10)%max * fp
vmin(1:10) = RA_stress(1:10)%min * fp
write(1234,format_(1:NNN)) 'max =   ',vmax(1:10)
write(1234,format_(1:NNN)) 'min=    ',vmin(1:10)
write(1234,format_(1:NNN)) 'max-min=',vmax(1:10)-vmin(1:10)
!0000000000000000000000000000000000000000000000000000
write(1234,*) ' !!! ******* TOTAL PRESSURE :  xx yy zz | tot | xy xz yz  : instant/RA/disp2/min/max   units=(1000 atm)'
write(1234,format_(1:NNN)) 'instant=',pressure(1:10) * fp
write(1234,format_(1:NNN)) 'averagd=',RA_pressure(1:10)%val/RA_pressure(1:10)%counts * fp
do i = 1,10 ; call disp2(RA_pressure(i),vct(i)); enddo
write(1234,format_(1:NNN)) 'disp2 = ',dsqrt(vct(1:10) ) * fp
vmax(1:10) = RA_pressure(1:10)%max * fp
vmin(1:10) = RA_pressure(1:10)%min * fp
write(1234,format_(1:NNN)) 'max =   ',vmax(1:10)
write(1234,format_(1:NNN)) 'min=    ',vmin(1:10)
write(1234,format_(1:NNN)) 'max-min=',vmax(1:10)-vmin(1:10)
write(1234,*) 'RA Stress details xx yy zz all xy xz yz : units = 1000atm'
write(1234,format_(1:NNN)) 'vdw ',RA1_stress_vdw(1:10)/RA1_stress_counts*fp
if (system_force_CTRL/=0) then
 write(1234,format_(1:NNN)) 'Qreal ',RA1_stress_Qreal(1:10)/RA1_stress_counts*fp
 write(1234,format_(1:NNN)) 'Qcmpl ',RA1_stress_Qcmplx(1:10)/RA1_stress_counts*fp
 if (i_boundary_CTRL ==1) then 
   write(1234,format_(1:NNN)) 'Qcmpl_k0 ',RA1_stress_Qcmplx_k_eq_0(1:10)/RA1_stress_counts*fp ! for 2D
   write(1234,format_(1:NNN)) 'Qcmpas3D ',RA1_stress_Qcmplx_as_in_3D(1:10)/RA1_stress_counts*fp
 endif
endif !system_force_CTRL/=0 (it has charges)
write(1234,format_(1:NNN)) '-excluded ',-RA1_stress_excluded(1:10)/RA1_stress_counts*fp
if(Nbonds>0)      write(1234,format_(1:NNN)) 'bond ',RA1_stress_bond(1:10)/RA1_stress_counts*fp
if(Nangles>0)     write(1234,format_(1:NNN)) 'angle ',RA1_stress_angle(1:10)/RA1_stress_counts*fp
if(Ndihedrals>0)  write(1234,format_(1:NNN)) 'dih ',RA1_stress_dih(1:10)/RA1_stress_counts*fp
if(Ndeforms>0)    write(1234,format_(1:NNN)) 'deform ',RA1_stress_deform(1:10)/RA1_stress_counts*fp
if(Nconstrains>0) write(1234,format_(1:NNN)) 'shake ',RA1_stress_shake(1:10)/RA1_stress_counts*fp
if(Ndummies>0)    write(1234,format_(1:NNN)) 'Qdummy ',RA1_stress_dummy(1:10)/RA1_stress_counts*fp
if(thetering%N>0) write(1234,format_(1:NNN)) 'thetering ',RA1_stress_thetering(1:10)/RA1_stress_counts*fp
write(1234,format_(1:NNN)) 'kinetic ',RA1_stress_kin(1:10)/RA1_stress_counts*fp
write(1234,*) '\\End stress info'

if (l_do_QN_CTRL) then
 write(1234,*) '!!! ******* TEMPERATURE trans rot |  instant/RA/disp2/min/max     units = Kelvin'
 write(1234,'(A8,2(F10.3,1X))') 'instant=',Temperature_trans_Calc,Temperature_rot_Calc 
 write(1234,'(A8,2(F10.3,1X))') 'averagd=',RA_Temperature_trans%val/RA_Temperature_trans%counts, &
RA_Temperature_rot%val/RA_Temperature_rot%counts
 call disp2(RA_temperature_trans,vct(1));
 call disp2(RA_temperature_rot,vct(2));
 write(1234,'(A8,2(F10.3,1X))') 'disp2 = ',dsqrt(vct(1:2))
 vmax(1) = RA_Temperature_trans%max   
 vmin(1) = RA_Temperature_trans%min
 vmax(1) = RA_Temperature_rot%max
 vmin(1) = RA_Temperature_rot%min
 write(1234,'(A8,2(F10.3,1X))') 'max =   ',vmax(1:2)
 write(1234,'(A8,2(F10.3,1X))') 'min=    ',vmin(1:2)
 write(1234,'(A8,2(F10.3,1X))') 'max-min=',vmax(1:2) - vmin(1:2)
else
 write(1234,*) '!!! ******* TEMPERATURE |  instant/RA/disp2/min/max     units = Kelvin'
 write(1234,'(A8,1(F10.3,1X))') 'instant=',T_eval
 write(1234,'(A8,1(F10.3,1X))') 'averagd=',RA_temperature%val/RA_temperature%counts 
 call disp2(RA_temperature,vct(1)); 
 write(1234,'(A8,1(F10.3,1X))') 'disp2 = ',dsqrt(vct(1:1)) 
 vmax(1) = RA_temperature%max 
 vmin(1) = RA_temperature%min 
 write(1234,'(A8,1(F10.3,1X))') 'max =   ',vmax(1)
 write(1234,'(A8,1(F10.3,1X))') 'min=    ',vmin(1)
 write(1234,'(A8,1(F10.3,1X))') 'max-min=',vmax(1) - vmin(1)
endif
!----------------------------------------------------
write(1234,*) ' !!! ******** LINEAR MOMENTUM CONSERVATION | xx yy zz:  instant/RA/disp2/min/max     units = (u.a.m*Amstrom/ps)'
if (l_do_QN_CTRL) then
write(1234,*) 'instant=',sum(mol_MOM(:,1)),sum(mol_MOM(:,2)), sum(mol_MOM(:,3))
else
write(1234,*) 'instant=',dot_product(vxx,all_atoms_mass),dot_product(vyy,all_atoms_mass),dot_product(vzz,all_atoms_mass)
endif
write(1234,*) 'averagd=',RA_MOM_0%val/RA_MOM_0%counts
do i = 1, 3 ;  call disp2(RA_MOM_0(i),vct(i)); enddo
write(1234,*) 'disp2 = ',dsqrt(vct(1:3))
vmax(1:3) = RA_MOM_0(1:3)%max
vmin(1:3) = RA_MOM_0(1:3)%min
write(1234,*) 'max =   ',vmax(1:3)
write(1234,*) 'min=    ',vmin(1:3)
write(1234,*) 'max-min=',vmax(1:3)-vmin(1:3)
!--------------------------------------------------
fq =  dsqrt( Red_Vacuum_EL_permitivity_4_Pi) ! 
if (l_ANY_QG_pol_CTRL.or.l_ANY_QP_pol_CTRL.or.l_ANY_SFIELD_CTRL) then
write(1234,*) ' !!!  ********* SUM OF ELECTRIC CHARGES | point/gauss/both:  instant/RA/disp2/min/max     units = electron'
write(1234,*) 'instant=',sum(all_p_charges)*fq, sum(all_g_charges)*fq, sum(all_p_charges+all_g_charges)*fq
write(1234,*) 'averagd=',RA_sum_charge%val/RA_sum_charge%counts*fq
do i = 1,3 ;  call disp2(RA_sum_charge(i),vct(i)); enddo 
write(1234,*) 'disp2 = ',dsqrt(vct(1:3)) * fq
vmax(1:3) = RA_sum_charge(1:3)%max * fq
vmin(1:3) = RA_sum_charge(1:3)%min * fq
write(1234,*) 'max =   ',vmax(1:3)
write(1234,*) 'min=    ',vmin(1:3)
write(1234,*) 'max-min=',vmax(1:3)-vmin(1:3)
endif
!------------------------------------------------
if (l_ANY_SFIELD_CTRL.or.l_ANY_POL_CTRL) then
write(1234,*) '!!!! *****  Iterative method(cg) convergence | iterations'
write(1234,*) 'instant=',int(cg_iterations)
write(1234,'(A8,F9.3)') 'averagd=',RA_cg_iter%val/RA_cg_iter%counts
call disp2(RA_cg_iter,vct(1));
write(1234,'(A8,F9.3)') 'disp2 = ',dsqrt(vct(1:1))
write(1234,'(A8,F9.3)') 'max =   ',RA_cg_iter%max
write(1234,'(A8,F9.3)') 'min=    ',RA_cg_iter%min 
write(1234,'(A8,F9.3)') 'max-min=',(RA_cg_iter%max -RA_cg_iter%min) 
endif
if (Nconstrains>0) then
write(1234,*) '!!!! *****  Iterative SHAKE convergency'
write(1234,*) 'average shake iters = ',RA_shake_iter%val/RA_shake_iter%counts
call disp2(RA_shake_iter,vct(1));
write(1234,'(A8,F9.3)') 'disp2 = ',dsqrt(vct(1:1))
write(1234,'(A8,F9.3)') 'max =   ',RA_shake_iter%max
write(1234,'(A8,F9.3)') 'min=    ',RA_shake_iter%min
write(1234,'(A8,F9.3)') 'max-min=',(RA_shake_iter%max -RA_shake_iter%min)
endif



!   type(statistics_5_type) RA_msd2(4)      ! mean square displacemets xx, yy, zz r
!   type(statistics_5_type) RA_diffusion(3) !x^2+y^2, z^2, r^2  ! based on atom positions.

    if (sys_prep%any_prep) then
      write(1234,*)  'WARNING : THIS JOB IS A SYSTEM PREPARATION JOB ONLY!!!!'
      if (sys_prep%type_prep==0) then
      write(1234,*)  'THE BOXES XX YY ZZ WILL BE ABJUSTED TO A FINAL VALUE OF : ',sys_prep%box_to(:)
      write(1234,*)  'CURRENT VALUE OF BOX: =',sim_cel(1),sim_cel(5),sim_cel(9)
      endif
      if (sys_prep%type_prep==1) then
      write(1234,*)  'THE ZZ position of sfc constrained atoms (like electrodes) ',&
              'WILL BE ABJUSTED BY AN INCREMENT OF : ',sys_prep%zsfc_by
      endif
      write(1234,*)  'IF you wish to run a production run then just comment the record',&
      ' SYSTEM_PREPARATION  from the line ',sys_prep%where_in_file, 'in the input file in.in'
      write(1234,*)  '\End of system preparation warning'
    endif

write(1234,*)'--------------------------------------------------------------------------------'
close(1234)

contains

 subroutine disp2(RA,value)
 use types_module, only : statistics_5_type
 implicit none
 type(statistics_5_type) , intent(IN) :: RA
 real(8) , intent(OUT) :: value
 real(8) avg
   avg = RA%val/RA%counts
   value = RA%val_sq/RA%counts - avg*avg
 end subroutine disp2

end subroutine write_scalar_statistics

subroutine write_history
use sizes_data, only : Natoms, Nmols,N_type_atoms
use mol_type_data, only : N_type_molecules
use history_data, only : history
use mol_type_data, only : N_type_atoms_per_mol_type
use integrate_data, only : time_step,integration_step,N_MD_STEPS
use sim_cel_data, only :  sim_cel,i_boundary_CTRL
use ALL_atoms_data, only :xxx,yyy,zzz,vxx,vyy,vzz,fxx,fyy,fzz,&
is_sfield_constrained,is_dipole_polarizable,&
all_charges,all_p_charges,all_g_charges,&
all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,all_dipoles
use file_names_data, only : path_out,continue_job,nf_history,nf_history_Q
use profiles_data, only : atom_profile,l_need_2nd_profile
use physical_constants, only : Red_Vacuum_EL_permitivity_4_Pi

implicit none
logical,save :: first_time = .true.
logical, save :: any_sfield =.false.
logical, save :: any_induced_dipole =.false.
logical, save :: any_
integer i
real(8) fq1

fq1=dsqrt(Red_Vacuum_EL_permitivity_4_Pi)
if (first_time) then
first_time=.false.
nf_history=trim(path_out)//'HISTORY_'//trim(continue_job%field1%ch)//'.BIN'
nf_history_Q=trim(path_out)//'HISTORY_Q_'//trim(continue_job%field1%ch)//'.BIN'
open(unit=33,file=trim(nf_history),status='replace',form='unformatted')
do i = 1, ubound(is_sfield_constrained,dim=1) ; 
 if (is_sfield_constrained(i) ) any_sfield=.true.
enddo
do i = 1, ubound(is_dipole_polarizable,dim=1)
 if (is_dipole_polarizable(i)) any_induced_dipole=.true.
enddo
any_=any_induced_dipole.or.any_sfield
if (any_) then
  open(unit=34,file=trim(nf_history_Q),status='replace',form='unformatted')
  write(34) Natoms,Nmols
  write(34) is_sfield_constrained(:)
  write(34) is_dipole_polarizable(:)
  close(34)
endif
write(33)Natoms,Nmols, N_type_atoms, N_type_molecules
write(33) N_type_atoms_per_mol_type
write(33) sim_cel,i_boundary_CTRL
write(33) history%how_often,history%cel,history%x,history%v,history%f,history%en
write(33) N_MD_STEPS/history%how_often
close(33)
endif

if (mod(integration_step,history%how_often)==0)then
open(unit=33,file=trim(nf_history),status='old',access='append',form='unformatted')
if (any_) open(unit=34,file=trim(nf_history_Q),status='old',access='append',form='unformatted')

write(33) integration_step,dble(integration_step)*time_step

if (any_) then
   write(34) integration_step,dble(integration_step)*time_step
   do i = 1, Natoms
     if(is_sfield_constrained(i)) write(34) all_charges(i)*fq1,all_p_charges(i)*fq1,all_g_charges(i) *fq1
   enddo
   do i = 1, Natoms
     if (is_dipole_polarizable(i)) write(34)  all_dipoles_xx(i)*fq1,all_dipoles_yy(i)*fq1,all_dipoles_zz(i)*fq1,all_dipoles(i)*fq1
   enddo
endif


if (history%cel==1) write(33) sim_cel,i_boundary_CTRL
if (history%x==1)    write(33) xxx,yyy,zzz
if (history%v==1)    write(33) vxx,vyy,vzz
if (history%f==1)    write(33) fxx,fyy,fzz
if (history%en==1.and.l_need_2nd_profile)   write(33) atom_profile%pot,(atom_profile%pot+atom_profile%kin)
close(33)
if(any_)close(34)
endif

end subroutine write_history

!subroutine read_history(nf)
!implicit none
!character(*), intent(IN)::nf
!integer i
!open(unit=33,file=trim(nf),form='unformatted')
!read(33)Natoms,Nmols, N_type_atoms, N_type_molecules
!close(33)
!allocate(N_type_atoms_per_mol_type(N_type_molecules))
!open(unit=33,file=trim(nf),form='unformatted')
!read(33)Natoms,Nmols, N_type_atoms, N_type_molecules,N_type_atoms_per_mol_type
!read(33) sim_cel_0(1:9),i_boundary_CTRL_0
!read(33) history%how_often,history%cel,history%x,history%v,history%f
!read(33) N_records
!allocate(xxx(Natoms),yyy(Natoms),zzz(Natoms))
!allocate(vxx(Natoms),vyy(Natoms),vzz(Natoms))
!allocate(fxx(Natoms),fyy(Natoms),fzz(Natoms))

!do i = 1, N_records
!read(33) integrations_step,dble(integrations_step)*time_step
!if (history%cel==1) read(33) sim_cel(1:9),i_boundary_CTRL
!if (history%x==1)   read(33) xxx,yyy,zzz
!if (history%v==1)   read(33) vxx,vyy,vzz
!if (history%f==1)   read(33) fxx,fyy,fzz
!close(33)
!enddo
!deallocate(xxx,yyy,zzz)
!deallocate(vxx,vyy,vzz)
!deallocate(fxx,fyy,fzz)
!end subroutine read_history

end module write_stuff
