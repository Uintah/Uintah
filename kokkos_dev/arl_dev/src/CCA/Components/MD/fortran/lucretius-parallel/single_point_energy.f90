
module single_point_energy_module

public :: intramolecular_enery_driver
public :: 
contains 
subroutine intramolecular_enery_driver
use sizes_data, only : Nbonds,Nangles,Ndihedrals,Ndeforms

    if(Nbonds>0)      call bond_energy
    if(Nangles>0)     call angle_energy
    if(Ndihedrals>0)  call dihedral_energy
    if(Ndeforms>0)    call out_of_plane_deforms_energy
end subroutine intramolecular_energy_driver

subroutine bond_energy
use connectivity_type_data, only : bond_types, prm_bond_types
use connectivity_ALL_data, only : list_bonds, is_bond_constrained,is_bond_dummy
use ALL_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz,Natoms
use sizes_data, only : Nbonds
use boundaries, only : periodic_images
use paralel_env_data
use stresses_data, only : stress_bond,stress
use energies_data, only : en_bond
use paralel_env_data
implicit none
integer i1,i,j,k,iat,jat, N, ibond, iStyle, itype
real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
real(8) a,b,En,ff,fx,fy,fz,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz, x
real(8) lst_xx,lst_yy,lst_zz,lst_xy,lst_xz,lst_yz, local_energy
logical execute

 N = Nbonds
 allocate(dx(N),dy(N),dz(N),dr_sq(N))
 local_energy = 0.0d0
 i1 = 0
 do ibond = 1+rank, Nbonds, nprocs
 execute = .not.(is_bond_constrained(ibond).or.is_bond_dummy(ibond))
 if (execute) then
 i1 = i1 + 1
  iat = list_bonds(1,ibond)
  jat = list_bonds(2,ibond)
  dx(i1) = xxx(iat) - xxx(jat)
  dy(i1) = yyy(iat) - yyy(jat)
  dz(i1) = zzz(iat) - zzz(jat)
 endif
 enddo
 call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
 dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)

 i1=0
 do  ibond = 1 + rank , Nbonds, nprocs
 execute = .not.(is_bond_constrained(ibond).or.is_bond_dummy(ibond))
 if (execute) then
 i1 = i1 + 1
 x = dsqrt(dr_sq(i1))
  iat    = list_bonds(1,ibond)
  jat    = list_bonds(2,ibond)
  itype  = list_bonds(0,ibond)
!  iStyle = bond_types(1,itype)
  iStyle = list_bonds(4,ibond)
  select case (iStyle)
  case(1)  ! Harmonic
    b = prm_bond_types(2,itype)   ;
    a = prm_bond_types(1,itype)
    En = 0.5d0*(a*(x-b))*(x-b)
  case default
      print*, 'NOT DEFINED Style of bond',istyle, ' in subroutine bond_singlepoint_energies'
      STOP
  end select

 local_energy = local_energy + En

 endif ! execute
 enddo   !  ibond = 1, Nbonds

 en_bond = local_energy
 deallocate(dx,dy,dz,dr_sq)
end subroutine bond_energy

subroutine angle_energy
use connectivity_type_data, only : angle_types, prm_angle_types
use connectivity_ALL_data, only : list_angles
use ALL_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz,xx,yy,zz
use boundaries, only : periodic_images
use sizes_data, only : Nangles
use energies_data, only : en_angle
use paralel_env_data
use stresses_data, only : stress_angle, stress
use paralel_env_data
implicit none
 integer i,j,k,iangle,iat,jat,kat,itype,iStyle,i1,N
 real(8), allocatable :: dx12(:),dy12(:),dz12(:),dx23(:),dy23(:),dz23(:)
 real(8), allocatable :: r12_sq(:),r23_sq(:)
 real(8) i_r12,i_r23,r12,r23,x12,y12,z12,x23,y22,y23,z23,fx1,fx2,fy1,fy2,fz1,fz2, fx3,fy3,fz3
 real(8) a0,AA,b,En,ff,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz, cos_123, sin_123, angle
 real(8) AA2,AA3,AA4,b2,b3,b4
 real(8) lst_xx,lst_yy,lst_zz,lst_xy,lst_xz,lst_yz, local_energy
 real(8), parameter :: inv3=1.0d0/3.0d0

 N = Nangles
 allocate(dx12(N),dy12(N),dz12(N))
 allocate(dx23(N),dy23(N),dz23(N))
 allocate(r12_sq(N),r23_sq(N))
 local_energy = 0.0d0
 i1 = 0
 do iangle = 1+rank, Nangles,nprocs
  i1 = i1 + 1
  iat = list_angles(1,iangle)  ;  jat = list_angles(2,iangle)  ;  kat = list_angles(3,iangle)
  dx12(i1) = xxx(iat)-xxx(jat)  ; dy12(i1) = yyy(iat)-yyy(jat)  ; dz12(i1) = zzz(iat)-zzz(jat)
  dx23(i1) = xxx(kat)-xxx(jat)  ; dy23(i1) = yyy(kat)-yyy(jat)  ; dz23(i1) = zzz(kat)-zzz(jat)
 enddo
 call periodic_images(dx12(1:i1),dy12(1:i1),dz12(1:i1))
 call periodic_images(dx23(1:i1),dy23(1:i1),dz23(1:i1))
 r12_sq(1:i1) = dx12(1:i1)* dx12(1:i1)+dy12(1:i1)*dy12(1:i1)+dz12(1:i1)*dz12(1:i1)
 r23_sq(1:i1) = dx23(1:i1)* dx23(1:i1)+dy23(1:i1)*dy23(1:i1)+dz23(1:i1)*dz23(1:i1)

 i1 = 0
 do iangle = 1+rank, Nangles, nprocs
  i1 = i1 + 1
  iat = list_angles(1,iangle)  ;  jat = list_angles(2,iangle)  ;  kat = list_angles(3,iangle)
  itype = list_angles(0,iangle)
!  iStyle = angle_types(1,itype)
  iStyle = list_angles(5,iangle)
  r12 = dsqrt(r12_sq(i1))  ;  r23 = dsqrt(r23_sq(i1))
  i_r12 = 1.0d0/r12        ; i_r23 = 1.0d0/r23
  x12 = dx12(i1)*i_r12  ; y12 = dy12(i1)*i_r12  ; z12 = dz12(i1)*i_r12
  x23 = dx23(i1)*i_r23  ; y23 = dy23(i1)*i_r23  ; z23 = dz23(i1)*i_r23
  cos_123 = x12*x23 + y12*y23 + z12*z23
  angle = dacos(cos_123)
  sin_123 = max(1.0d-9,dsqrt(max(0.0d0,(1.0d0-cos_123*cos_123))))

  select case (iStyle)
    case(1)  ! harmonic with respect to angle
     a0 = prm_angle_types(2,itype)  ! the equilibrium angle
     b = prm_angle_types(1,itype)
     En = 0.5d0*(b*(angle-a0))*(angle-a0)
    case(2)  ! Anarmonic order 4
      a0 = prm_angle_types(4,itype)  ! the equilibrium angle
      b2 = prm_angle_types(1,iangle)
      b3 = prm_angle_types(2,iangle)
      b4 = prm_angle_types(3,iangle)
      AA = (angle-a0)
      AA2 = AA*AA ; AA3 = AA*AA2 ; AA4 = AA2*AA2
      En =  0.5d0*b2*AA2 + inv3*b3*AA3 + 0.25d0*b4*AA4
    case default
  end select
  local_energy = local_energy + En
 enddo ! iangle = 1, Nangles

 en_angle = local_energy

 deallocate(r12_sq,r23_sq)
 deallocate(dx23,dy23,dz23)
 deallocate(dx12,dy12,dz12)

end subroutine angle_energy

subroutine dihedral_energy
use connectivity_type_data, only : dihedral_types, prm_dihedral_types, Nfolds_dihedral_types
use connectivity_ALL_data, only : list_dihedrals
use ALL_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz
use boundaries, only : periodic_images
use sizes_data, only : Ndihedrals
use energies_data, only : en_dih
use paralel_env_data
use stresses_data, only : stress_dih,stress
use math_constants, only : Pi2
use paralel_env_data
implicit none
 integer i,j,k,l,iat,jat,kat,iStyle,itype,i1, idihedral,N
 real(8), allocatable :: dx12(:),dy12(:),dz12(:),dx23(:),dy23(:),dz23(:),dx34(:),dy34(:),dz34(:)
 real(8), allocatable :: r12_sq(:),r23_sq(:),r34_sq(:)
 real(8) a0,AA,b,En,ff,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz, cos_dihedral, sin_dihedral, angle,sin_inv
 real(8) fct1,fct2,u2,t2,ut,fx1,fx2,fx3,fx4,fy1,fy2,fy3,fy4,fz1,fz2,fz3,fz4,fx23,fy23,fz23
 real(8) x12,y12,z12,x23,y23,z23,x34,y34,z34,ux,uy,uz,tx,ty,tz,i_u2,i_t2
 real(8) lst_xx,lst_yy,lst_zz,lst_xy,lst_xz,lst_yz, local_energy
 real(8) i_sqrt_t2_u2, i_pi2, phase
 integer N_folds
 integer itemp
integer ii,jj,kk,ll
 i_pi2=1.0d0/Pi2
 N = Ndihedrals
 allocate(dx12(N),dy12(N),dz12(N))
 allocate(dx23(N),dy23(N),dz23(N))
 allocate(dx34(N),dy34(N),dz34(N))
 allocate(r12_sq(N),r23_sq(N),r34_sq(N))
 local_energy = 0.0d0
 i1 = 0
 do idihedral = 1+rank, Ndihedrals,nprocs
  i1 = i1 + 1
  i = list_dihedrals(1,idihedral) ; j = list_dihedrals(2,idihedral)
  k = list_dihedrals(3,idihedral) ; l = list_dihedrals(4,idihedral)
  dx12(i1)=xxx(i)-xxx(j)  ;  dy12(i1)=yyy(i)-yyy(j)  ;  dz12(i1)=zzz(i)-zzz(j)
  dx23(i1)=xxx(j)-xxx(k)  ;  dy23(i1)=yyy(j)-yyy(k)  ;  dz23(i1)=zzz(j)-zzz(k)
  dx34(i1)=xxx(k)-xxx(l)  ;  dy34(i1)=yyy(k)-yyy(l)  ;  dz34(i1)=zzz(k)-zzz(l)
 enddo
 call periodic_images(dx12(1:i1),dy12(1:i1),dz12(1:i1))
 call periodic_images(dx23(1:i1),dy23(1:i1),dz23(1:i1))
 call periodic_images(dx34(1:i1),dy34(1:i1),dz34(1:i1))
 r12_sq(1:i1) = dx12(1:i1)* dx12(1:i1)+dy12(1:i1)*dy12(1:i1)+dz12(1:i1)*dz12(1:i1)
 r23_sq(1:i1) = dx23(1:i1)* dx23(1:i1)+dy23(1:i1)*dy23(1:i1)+dz23(1:i1)*dz23(1:i1)
 r34_sq(1:i1) = dx34(1:i1)* dx34(1:i1)+dy34(1:i1)*dy34(1:i1)+dz34(1:i1)*dz34(1:i1)
itemp=0

 i1 = 0
 do idihedral = 1+rank, Ndihedrals , nprocs
  i1 = i1 + 1
  x12 = dx12(i1) ; y12 = dy12(i1) ; z12 = dz12(i1)
  x23 = dx23(i1) ; y23 = dy23(i1) ; z23 = dz23(i1)
  x34 = dx34(i1) ; y34 = dy34(i1) ; z34 = dz34(i1) ! extract them in scalars for faster memory acces
  tx = y12*z23 - y23*z12  ;  ty = z12*x23 - z23*x12  ;  tz = x12*y23 - x23*y12
  t2 = tx*tx + ty*ty + tz*tz ; i_t2 = 1.0d0/t2
  ux = y23*z34 - y34*z23  ;  uy = z23*x34 - z34*x23  ;  uz = x23*y34 - x34*y23
  u2 = ux*ux + uy*uy + uz*uz ; i_u2 = 1.0d0/u2
  ut = tx*ux + ty*uy + tz*uz
      i_sqrt_t2_u2 = 1.0d0/dsqrt( t2 * u2 )
      cos_dihedral = ut * i_sqrt_t2_u2
      sin_dihedral = (x23*(tz*uy-uz*ty) - y23*(tz*ux-tx*uz) + z23*(ty*ux-tx*uy) ) / dsqrt(t2*u2*r23_sq(i1))
      angle = datan2(sin_dihedral,cos_dihedral)
      sin_dihedral=dsign(max(1.0d-9,dabs(sin_dihedral)),sin_dihedral)  ! avoid singularity
      sin_inv = 1.0d0/sin_dihedral

  ii = list_dihedrals(1,idihedral) ; jj = list_dihedrals(2,idihedral)
  kk = list_dihedrals(3,idihedral) ; ll = list_dihedrals(4,idihedral)
  itype = list_dihedrals(0,idihedral)
!  iStyle = dihedral_types(1,itype)
  iStyle = list_dihedrals(6,idihedral)
  phase = prm_dihedral_types(0,itype)
  select case (iStyle)
    case(1)

      N_folds = Nfolds_dihedral_types(itype) - 1
      En = prm_dihedral_types(1,itype) ;
      ff = 0.0d0 ;
      angle = angle-phase
      do i = 1, N_folds
         b = prm_dihedral_types(i+1,itype)
         En = En - b*(dcos(dble(i)*angle )   )
      enddo
    case(2) ! improper

      angle = angle-phase
      b = prm_dihedral_types(1,itype)
!      angle = angle-Pi2*(dble(int(2.0d0*(angle*i_pi2)))-dble(int(angle*i_pi2))     )   ! recenter it
      En = 0.5d0*b*angle*angle
    case default
      print*, ' DIHEDRAL CASE NOT DEFINED IN dihedral_forces',iStyle
      STOP
  end select
 local_energy = local_energy + En
 enddo
 en_dih = local_energy
 deallocate(r12_sq,r23_sq,r34_sq)
 deallocate(dx34,dy34,dz34)
 deallocate(dx23,dy23,dz23)
 deallocate(dx12,dy12,dz12)

 end subroutine dihedral_energy 

 subroutine out_of_plane_deforms_energy
use connectivity_type_data, only : deform_types, prm_deform_types
use connectivity_ALL_data, only : list_deforms
use ALL_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz,xx,yy,zz
use boundaries, only : periodic_images
use sizes_data, only : Ndeforms
use energies_data, only : en_deform
use paralel_env_data
use stresses_data, only : stress_deform,stress
use math_constants, only : Pi2
use paralel_env_data
implicit none
integer i_deforms
integer i,j,k,l,iat,jat,kat,iStyle,itype,i1, N
 real(8), allocatable :: dx12(:),dy12(:),dz12(:),dx23(:),dy23(:),dz23(:),dx24(:),dy24(:),dz24(:)
 real(8), allocatable :: r12_sq(:),r23_sq(:),r24_sq(:)
 real(8) a0,AA,b,En,ff,sxx,sxy,sxz,syx,syy,syz,szx,szy,szz, cos_dihedral, sin_dihedral, angle,sin_inv
 real(8) x24,y24,z24
 real(8) fct1,fct2,u2,t2,ut,fx1,fx2,fx3,fx4,fy1,fy2,fy3,fy4,fz1,fz2,fz3,fz4,fx23,fy23,fz23
 real(8) x12,y12,z12,x23,y23,z23,x34,y34,z34,ux,uy,uz,tx,ty,tz,i_u2,i_t2
 real(8) x13,y13,z13,x14,y14,z14
 real(8) lst_xx,lst_yy,lst_zz,lst_xy,lst_xz,lst_yz, lst_yx,lst_zx,lst_zy,local_energy
 real(8) pos1_xx,pos1_yy,pos1_zz,pos2_xx,pos2_yy,pos2_zz,pos3_xx,pos3_yy,pos3_zz,pos4_xx,pos4_yy,pos4_zz
 real(8) i_sqrt_t2_u2, i_pi2, phase
 integer N_folds
 integer itemp
integer ii,jj,kk,ll
 i_pi2=1.0d0/Pi2

 N = Ndeforms

 allocate(dx12(N),dy12(N),dz12(N))
 allocate(dx23(N),dy23(N),dz23(N))
 allocate(dx24(N),dy24(N),dz24(N))
 allocate(r12_sq(N),r23_sq(N),r24_sq(N))
 i1 = 0
 do i_deforms = 1+rank, Ndeforms, nprocs
  i1 = i1 + 1
  i = list_deforms(1,i_deforms) ; j = list_deforms(2,i_deforms)
  k = list_deforms(3,i_deforms) ; l = list_deforms(4,i_deforms)
  dx12(i1)=xxx(i)-xxx(j)  ;  dy12(i1)=yyy(i)-yyy(j)  ;  dz12(i1)=zzz(i)-zzz(j)
  dx23(i1)=xxx(j)-xxx(k)  ;  dy23(i1)=yyy(j)-yyy(k)  ;  dz23(i1)=zzz(j)-zzz(k)
  dx24(i1)=xxx(j)-xxx(l)  ;  dy24(i1)=yyy(j)-yyy(l)  ;  dz24(i1)=zzz(j)-zzz(l)
 enddo
 call periodic_images(dx12(1:i1),dy12(1:i1),dz12(1:i1))
 call periodic_images(dx23(1:i1),dy23(1:i1),dz23(1:i1))
 call periodic_images(dx24(1:i1),dy24(1:i1),dz24(1:i1))
 r12_sq(1:i1) = dx12(1:i1)* dx12(1:i1)+dy12(1:i1)*dy12(1:i1)+dz12(1:i1)*dz12(1:i1)
 r23_sq(1:i1) = dx23(1:i1)* dx23(1:i1)+dy23(1:i1)*dy23(1:i1)+dz23(1:i1)*dz23(1:i1)
 r24_sq(1:i1) = dx24(1:i1)* dx24(1:i1)+dy24(1:i1)*dy24(1:i1)+dz24(1:i1)*dz24(1:i1)

call initialize
itemp=0

  i1 = 0
  do i_deforms = 1+rank, Ndeforms, nprocs
    i1 = i1 + 1
    x12 = dx12(i1) ; y12 = dy12(i1) ; z12 = dz12(i1)
    x23 = dx23(i1) ; y23 = dy23(i1) ; z23 = dz23(i1)
    x24 = dx24(i1) ; y24 = dy24(i1) ; z24 = dz24(i1) ! extract them in scalars for faster memory acces

    ii = list_deforms(1,i_deforms) ; jj = list_deforms(2,i_deforms)
    kk = list_deforms(3,i_deforms) ; ll = list_deforms(4,i_deforms)
    itype  = list_deforms(0,i_deforms)
    iStyle = list_deforms(6,i_deforms)

    select case (iStyle)
      case (1) ! SMITH
      tx = y12*z23 - y23*z12  ;  ty = z12*x23 - z23*x12  ;  tz = x12*y23 - x23*y12
      t2 = tx*tx + ty*ty + tz*tz ; i_t2 = 1.0d0/t2
      ux = x24 ; uy = y24 ; uz = z24 ! it will not go
      u2 = ux*ux + uy*uy + uz*uz ; i_u2 = 1.0d0/u2
      ut = tx*ux + ty*uy + tz*uz
      i_sqrt_t2_u2 = dsqrt( i_t2 * i_u2 )
      sin_dihedral = - ut*i_sqrt_t2_u2
      sin_dihedral=dsign(max(1.0d-9,dabs(sin_dihedral)),sin_dihedral)
      sin_inv = 1.0d0/sin_dihedral
      cos_dihedral = max(1.0d-9,dsqrt(max(0.0d0,(1.0d0-sin_dihedral*sin_dihedral))))
      angle = dasin(sin_dihedral)
      phase = prm_deform_types(2,itype) ! zero
      angle = angle-phase
      b = prm_deform_types(1,itype)
      En = 0.5d0*b*angle*angle
    end select

   local_energy = local_energy + En
  enddo

   en_deform = local_energy

end end subroutine out_of_plane_deforms_energy


!------------------
subroutine pair_short_forces_Q
 use sys_data
 use paralel_env_data
 use math_constants
 use boundaries
 use ALL_atoms_data
 use atom_type_data
 use max_sizes_data, only : MX_list_nonbonded
 use non_bonded_lists_data, only : list_nonbonded, size_list_nonbonded
 use profiles_data
 use energies_data
 use stresses_data
 use atom_type_data
 use interpolate_data
 use variables_short_pairs
 use physical_constants, only : Volt_to_internal_field
 use rdfs_data, only : rdfs
 use integrate_data, only : integration_step
 use rdfs_collect_module, only : rdfs_collect
 use CTRLs_data, only : l_ANY_DIPOLE_CTRL, l_DIP_CTRL
 use compute_14_module, only : compute_14_interactions_driver
 use REAL_part_intra_correction_sfc_sfc
 implicit none
 real(8), parameter :: en_factor = 0.5d0
 integer i,j,i1,imol,itype,N,neightot,k,nneigh, iStyle
 real(8) local_energy,En0
 real(8), allocatable :: local_force(:,:)

! allocate(local_force(Natoms,3)) ;
! local_force(:,1) = fxx(:); local_force(:,2) = fyy(:); local_force(:,3) = fzz(:)
 en_vdw = 0.0d0
 en_Qreal = 0.0d0
 irdr = 1.0d0/rdr

 do i = 1+rank, Natoms , nprocs
  imol = atom_in_which_molecule(i)
  iStyle  = i_Style_atom(i)
  i1 = 0
  do k =  1, size_list_nonbonded(i)
    i1 = i1 + 1
    j = list_nonbonded(i,k)
    dx(i1) = xxx(i) - xxx(j)
    dy(i1) = yyy(i) - yyy(j)
    dz(i1) = zzz(i) - zzz(j)
    in_list_Q(i1) = j
  enddo
  neightot=i1

  call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
  dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
  neightot = i1
!  call nonbonded_vdw_2_forces(i,iStyle,size_list_nonbonded(i)) ! put it back

    do k =  1, neightot
    j = list_nonbonded(i,k)
    r2 = dr_sq(k)
    if ( r2 < cut_off_sq ) then
     jstyle = i_style_atom(j)
     i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
     a0 = atom_Style2_vdwPrm(0,i_pair)
     if (a0 > SYS_ZERO) then
        r = dsqrt(r2)
        Inverse_r_squared = 1.0d0/r2
        NDX = max(1,int(r*irdr))
        ppp = (r*irdr) - dble(ndx)
        vk  = vvdw(ndx,i_pair)  ;  vk1 = vvdw(ndx+1,i_pair) ; vk2 = vvdw(ndx+2,i_pair)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        En = (t1 + (t2-t1)*ppp*0.5d0)
        en_vdw = en_vdw + En
      endif ! (a0 > 1.0d-10
   endif
  enddo ! j index of the double loop
 
  qi = all_charges(i) ! electricity = charge no matter of what kind.
  dipole_xx_i = all_dipoles_xx(i) ; dipole_yy_i = all_dipoles_yy(i) ; dipole_zz_i = all_dipoles_zz(i)
  dipole_i2 = dipole_xx_i*dipole_xx_i+dipole_yy_i*dipole_yy_i+dipole_zz_i*dipole_zz_i
  call Qinner_initialize(i)
  if (l_DIP_CTRL) then

    i_displacement = 1.0d0/displacement
    do k =  1, neightot
    j = in_list_Q(k)
    r2 = dr_sq(k)
    if ( r2 < cut_off_sq ) then
     jStyle = i_Style_atom(j)


     i_pair = which_atomStyle_pair(iStyle,jStyle) ! can replace it by a formula?
     r = dsqrt(r2)
     Inverse_r = 1.0d0/r
     NDX = max(1,int(r*irdr))
     ppp = (r*irdr) - dble(ndx)
     x = dx(k)   ;  y = dy(k)    ; z = dz(k)
        qj = all_charges(j)
        dipole_xx_j = all_dipoles_xx(j) ; dipole_yy_j=all_dipoles_yy(j); dipole_zz_j=all_dipoles_zz(j)
        dipole_i_times_Rij = x*dipole_xx_i+y*dipole_yy_i+z*dipole_zz_i
        dipole_j_times_Rij = x*dipole_xx_j+y*dipole_yy_j+z*dipole_zz_j
        pipj = dipole_xx_i*dipole_xx_j + dipole_yy_i*dipole_yy_j+ dipole_zz_i*dipole_zz_j
        didj = dipole_i_times_Rij*dipole_j_times_Rij
        G1 = - dipole_i_times_Rij*qj + dipole_j_times_Rij*qi + pipj
        G2 = - didj
        qij = qi*qj
        include 'interpolate_4.frg'
        include 'interpolate_THOLE_ALL.frg'

        En =  B0*qij + B1*G1 + B2*G2 +    B0_THOLE*pipj + B1_THOLE*G2
        En_Qreal = En_Qreal + En
      endif ! if ( r2 < cut_off_sq )
    enddo ! k =  1, neightot

  else
!    call Q_2_forces(i,iStyle,neightot)
      i_displacement=1.0d0/displacement
  do k =  1, neightot
    j = in_list_Q(k)
    r2 = dr_sq(k)
    if ( r2 < cut_off_sq ) then
     jstyle = i_style_atom(j)
     i_pair = which_atomStyle_pair(istyle,jstyle) ! can replace it by a formula?
     r = dsqrt(r2)
     Inverse_r = 1.0d0/r
     NDX = max(1,int(r*irdr))
     ppp = (r*irdr) - dble(ndx)
     x = dx(k)   ;  y = dy(k)    ; z = dz(k)

        qj = all_charges(j)
        qij = qi*qj
        vk  = vele_G(ndx,i_pair)  ;  vk1 = vele_G(ndx+1,i_pair) ; vk2 = vele_G(ndx+2,i_pair)
        t1 = vk  + (vk1 - vk )*ppp
        t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
        En0 = (t1 + (t2-t1)*ppp*0.5d0)
        En = En0 * qij
        En_Qreal = En_Qreal + En
      endif
    enddo
  endif

  enddo ! i 

  call compute_14_interactions_driver
  call REAL_part_intra_correction_sfc_sfc_driver
local_energy = 0.0d0
do i = 1, Natoms
if(is_dipole_polarizable(i))then
  En0 = all_DIPOLE_pol(i)*(all_dipoles_xx(i)**2+all_dipoles_yy(i)**2+all_dipoles_zz(i)**2) * 0.5d0
  local_energy = local_energy + En0
  if (l_need_2nd_profile)then
    a_pot_Q(i) = a_pot_Q(i) + En0 * 2.0d0
  endif
endif
enddo
en_Qreal = en_Qreal + local_energy

  call finalize_scalar_props
  call finalize

!open(unit=14,file='fort.14',recl=222)
!do i = 1, Natoms
!write(14,*) i,fxx(i)/418.4d0,fyy(i)/418.4d0,fzz(i)/418.4d0
!enddo
!print*, 'stress=',stress_xx/418.4d0,stress_yy/418.4d0,stress_zz/418.4d0,&
!stress_xy/418.4d0,stress_xz/418.4d0,stress_yz/418.4d0
!write(14,*)'enQreal=',(en_Qreal+en_vdw)/418.4d0, en_Qreal/418.4d0, en_vdw/418.4d0,local_energy/418.4d0
!stop
 deallocate(dr_sq)
 deallocate(dx,dy,dz)
 deallocate(in_list_Q)
! deallocate(local_force)
   
end module single_point_energy_module


