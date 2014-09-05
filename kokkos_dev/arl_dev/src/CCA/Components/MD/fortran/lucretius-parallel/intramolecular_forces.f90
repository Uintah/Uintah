
module intramolecular_forces
!use boundaries

implicit none

public :: intramolecular_forces_driver
public :: intramolecular_driver_ENERGY
public :: intramolecular_driver_ENERGY_MCmove_1atom
public :: bond_forces
public :: angle_forces
public :: dihedral_forces
public :: out_of_plane_deforms
public :: bond_ENERGY
public :: angle_ENERGY
public :: dihedral_ENERGY
public :: out_of_plane_deforms_ENERGY
public :: bond_ENERGY_MCmove_1atom
public :: angle_ENERGY_MCmove_1atom
public :: dihedral_ENERGY_MCmove_1atom
public :: out_of_plane_deforms_ENERGY_MCmove_1atom
contains

subroutine intramolecular_forces_driver
use sizes_data, only : Nbonds,Nangles,Ndihedrals,Ndeforms

    if(Nbonds>0)      call bond_forces
    if(Nangles>0)     call angle_forces
    if(Ndihedrals>0)  call dihedral_forces
    if(Ndeforms>0)    call out_of_plane_deforms 
end subroutine intramolecular_forces_driver

!-------------------
subroutine intramolecular_driver_ENERGY
use sizes_data, only : Nbonds,Nangles,Ndihedrals,Ndeforms

    if(Nbonds>0)      call bond_ENERGY
    if(Nangles>0)     call angle_ENERGY
    if(Ndihedrals>0)  call dihedral_ENERGY
    if(Ndeforms>0)    call out_of_plane_deforms_ENERGY
end subroutine intramolecular_driver_ENERGY
!-------------------------
subroutine intramolecular_driver_ENERGY_MCmove_1atom(iwhich)
use sizes_data, only : Nbonds,Nangles,Ndihedrals,Ndeforms
implicit none
integer,intent(IN):: iwhich

    if(Nbonds>0)      call bond_ENERGY_MCmove_1atom(iwhich)
    if(Nangles>0)     call angle_ENERGY_MCmove_1atom(iwhich)
    if(Ndihedrals>0)  call dihedral_ENERGY_MCmove_1atom(iwhich)
    if(Ndeforms>0)    call out_of_plane_deforms_ENERGY_MCmove_1atom(iwhich)
end subroutine intramolecular_driver_ENERGY_MCmove_1atom


!--------------------------
subroutine bond_forces
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
 call initialize
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
    ff = -(a*(x-b)) / x  !(dE/dx)*1/x)
  case default
      print*, 'NOT DEFINED Style of bond',istyle, ' in subroutine bond_forces'
      STOP
  end select    

 local_energy = local_energy + En

 fx = ff*dx(i1)   ;   fy = ff*dy(i1)   ;  fz = ff*dz(i1)
 fxx(iat) = fxx(iat) + fx  ;  fxx(jat) = fxx(jat) - fx
 fyy(iat) = fyy(iat) + fy  ;  fyy(jat) = fyy(jat) - fy
 fzz(iat) = fzz(iat) + fz  ;  fzz(jat) = fzz(jat) - fz
 sxx = fx*dx(i1)  ; sxy = fx*dy(i1) ; sxz = fx*dz(i1) ; syy = fy*dy(i1) ; syz = fy*dz(i1) ; szz = fz*dz(i1)
 lst_xx = lst_xx + sxx; 
 lst_xy = lst_xy + sxy; 
 lst_xz = lst_xz + sxz
 lst_yy = lst_yy + syy; 
 lst_zz = lst_zz + szz 
 lst_yz = lst_yz + syz


! if (l_need_2nd_profile) then
!  call increment_profiles(bond_profile(ibond), En,sxx,sxy,sxz,syy,syz,szz,fxx,fxy,fxz)
! endif

 endif ! execute
 enddo   !  ibond = 1, Nbonds

 en_bond = local_energy
 stress_bond(1) = lst_xx
 stress_bond(2) = lst_yy
 stress_bond(3) = lst_zz
 stress_bond(4) = (lst_xx+lst_yy+lst_zz)/3.0d0
 stress_bond(5) = lst_xy
 stress_bond(6) = lst_xz
 stress_bond(7) = lst_yz
 stress_bond(8) = lst_xy
 stress_bond(9) = lst_xz
 stress_bond(10)= lst_yz
 stress(:) = stress(:) + stress_bond(:) 
 deallocate(dx,dy,dz,dr_sq)

!print*,'en_angle=',en_bond/100.0d0/4.184d0
!print*, 'stress_angle=',stress_bond/418.4d0
!do i = 1, ubound(fxx,dim=1)
!write(14,'(I6,1X,3(F14.7,1X))') i,fxx(i)/100.0d0/4.184d0,fyy(i)/100.0d0/4.184d0,fzz(i)/100.0d0/4.184d0
!enddo

 contains 
 subroutine initialize
   local_energy = 0.0d0
   lst_xx = 0.0d0
   lst_xy = 0.0d0
   lst_xz = 0.0d0
   lst_yy = 0.0d0
   lst_zz = 0.0d0
   lst_yz = 0.0d0

!   if (l_need_2nd_profile) then
!    do i = 1+rank, Nbonds,nprocs
!      call zero_atom_profile(bond_profile(i))
!    enddo
!    endif
 end subroutine initialize
end subroutine bond_forces

!------------------
subroutine bond_ENERGY
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
end subroutine bond_ENERGY

!-----------------

subroutine bond_ENERGY_MCmove_1atom(iwhich)
use connectivity_type_data, only : bond_types, prm_bond_types
use connectivity_ALL_data, only : list_bonds, is_bond_constrained,is_bond_dummy
use ALL_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz,Natoms
use sizes_data, only : Nbonds
use boundaries, only : periodic_images
use paralel_env_data
use stresses_data, only : stress_bond,stress
use d_energies_data, only : d_en_bond
use paralel_env_data
implicit none
integer, intent(IN) :: iwhich
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
  iat = list_bonds(1,ibond)
  jat = list_bonds(2,ibond)
  if (iat==iwhich.or.jat==iwhich) then
     i1 = i1 + 1
     dx(i1) = xxx(iat) - xxx(jat)
     dy(i1) = yyy(iat) - yyy(jat)
     dz(i1) = zzz(iat) - zzz(jat)
  endif
 endif
 enddo
 if (i1>0)then
 call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
 dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)

 i1=0
 do  ibond = 1 + rank , Nbonds, nprocs
 execute = .not.(is_bond_constrained(ibond).or.is_bond_dummy(ibond))
 if (execute) then
  iat    = list_bonds(1,ibond)
  jat    = list_bonds(2,ibond)
  if (iat==iwhich.or.jat==iwhich) then
     i1 = i1 + 1
     x = dsqrt(dr_sq(i1))
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
   endif !iat==iwhich.or.jat==iwhich
 endif ! execute
 enddo   !  ibond = 1, Nbonds
 endif ! i1>0
 d_en_bond = local_energy
 deallocate(dx,dy,dz,dr_sq)
end subroutine bond_ENERGY_MCmove_1atom

!---------------------
!--------------------

!---NOW ANGLES -----------
subroutine angle_forces
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
 call initialize
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
     ff = (b*(angle-a0)) / sin_123 
     fx1=ff*(x23-x12*cos_123)*i_r12 ; fy1=ff*(y23-y12*cos_123)*i_r12 ; fz1 = ff*(z23-z12*cos_123)*i_r12
     fx3=ff*(x12-x23*cos_123)*i_r23 ; fy3=ff*(y12-y23*cos_123)*i_r23 ; fz3 = ff*(z12-z23*cos_123)*i_r23
    case(2)  ! Anarmonic order 4
      a0 = prm_angle_types(4,itype)  ! the equilibrium angle
      b2 = prm_angle_types(1,iangle)
      b3 = prm_angle_types(2,iangle)
      b4 = prm_angle_types(3,iangle)
      AA = (angle-a0)
      AA2 = AA*AA ; AA3 = AA*AA2 ; AA4 = AA2*AA2
      En =  0.5d0*b2*AA2 + inv3*b3*AA3 + 0.25d0*b4*AA4
      ff = (b2*AA +  b3*AA2 + b4*AA3 )  / sin_123
      fx1=ff*(x23-x12*cos_123)*i_r12 ; fy1=ff*(y23-y12*cos_123)*i_r12 ; fz1 = ff*(z23-z12*cos_123)*i_r12
      fx3=ff*(x12-x23*cos_123)*i_r23 ; fy3=ff*(y12-y23*cos_123)*i_r23 ; fz3 = ff*(z12-z23*cos_123)*i_r23

    case default
  end select

!write(14,*) iangle,iat,jat,kat
!write(14,'(I5,1X,3(F14.7,1X))')iangle,xx(iat),yy(iat),zz(iat)
!write(14,'(I5,1X,3(F14.7,1X))')iangle,xx(jat),yy(jat),zz(jat)
!write(14,'(I5,1X,3(F14.7,1X))')iangle,xx(kat),yy(kat),zz(kat)
!
!write(14,'(I5,1X,3(F14.7,1X))')iangle,fx1/418.4d0,fy1/418.4d0,fz1/418.4d0
!write(14,'(I5,1X,3(F14.7,1X))')iangle,fx3/418.4d0,fy3/418.4d0,fz3/418.4d0

  local_energy = local_energy + En
  fxx(iat)=fxx(iat)+fx1
  fyy(iat)=fyy(iat)+fy1
  fzz(iat)=fzz(iat)+fz1
  fxx(jat)=fxx(jat)-(fx1+fx3)
  fyy(jat)=fyy(jat)-(fy1+fy3)
  fzz(jat)=fzz(jat)-(fz1+fz3)
  fxx(kat)=fxx(kat)+fx3
  fyy(kat)=fyy(kat)+fy3
  fzz(kat)=fzz(kat)+fz3

 sxx = r12*x12*fx1+r23*x23*fx3  ; sxy = r12*x12*fy1+r23*x23*fy3 ; sxz = r12*x12*fz1+r23*x23*fz3
       syy = r12*y12*fy1+r23*y23*fy3 ; syz = r12*y12*fz1+r23*y23*fz3 ; szz = r12*z12*fz1+r23*z23*fz3
 
   lst_xx = lst_xx + sxx
   lst_xy = lst_xy + sxy
   lst_xz = lst_xz + sxz
   lst_yy = lst_yy + syy
   lst_zz = lst_zz + szz
   lst_yz = lst_yz + syz

! if (l_need_2nd_profile) then
!  call increment_profiles(angle_profile(iangle), En,sxx,sxy,sxz,syy,syz,szz,fxx,fxy,fxz)
! endif
 enddo ! iangle = 1, Nangles 

 en_angle = local_energy 
 stress_angle(1) = lst_xx
 stress_angle(2) = lst_yy
 stress_angle(3) = lst_zz
 stress_angle(4) = (lst_xx+lst_yy+lst_zz)/3.0d0
 stress_angle(5) = lst_xy
 stress_angle(6) = lst_xz
 stress_angle(7) = lst_yz
 stress_angle(8) = lst_xy
 stress_angle(9) = lst_xz
 stress_angle(10)= lst_yz
 stress = stress + stress_angle

!print*,'en_angle=',en_angle/100.0d0/4.184d0
!print*, 'stress_angle=',stress_angle/418.4d0
!do i = 1, ubound(fxx,dim=1)
!write(14,'(I6,1X,3(F14.7,1X))') i,fxx(i)/100.0d0/4.184d0,fyy(i)/100.0d0/4.184d0,fzz(i)/100.0d0/4.184d0
!enddo

 deallocate(r12_sq,r23_sq)
 deallocate(dx23,dy23,dz23)
 deallocate(dx12,dy12,dz12)
 contains
 subroutine initialize
   local_energy = 0.0d0
   lst_xx = 0.0d0
   lst_xy = 0.0d0
   lst_xz = 0.0d0
   lst_yy = 0.0d0
   lst_zz = 0.0d0
   lst_yz = 0.0d0

!   if (l_need_2nd_profile) then
!    do i = 1+rank, Nangles,nprocs
!      call zero_atom_profile(angle_profile(i))
!    enddo
!    endif
 end subroutine initialize

end subroutine angle_forces
!------------------

subroutine angle_ENERGY
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

end subroutine angle_ENERGY

!--------------------

subroutine angle_ENERGY_MCmove_1atom(iwhich)
use connectivity_type_data, only : angle_types, prm_angle_types
use connectivity_ALL_data, only : list_angles
use ALL_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz,xx,yy,zz
use boundaries, only : periodic_images
use sizes_data, only : Nangles
use d_energies_data, only : d_en_angle
use paralel_env_data
use stresses_data, only : stress_angle, stress
use paralel_env_data
implicit none
integer, intent(IN)  :: iwhich
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
  iat = list_angles(1,iangle)  ;  jat = list_angles(2,iangle)  ;  kat = list_angles(3,iangle)
  if (iat==iwhich.or.jat==iwhich.or.kat==iwhich) then
    i1 = i1 + 1
   dx12(i1) = xxx(iat)-xxx(jat)  ; dy12(i1) = yyy(iat)-yyy(jat)  ; dz12(i1) = zzz(iat)-zzz(jat)
   dx23(i1) = xxx(kat)-xxx(jat)  ; dy23(i1) = yyy(kat)-yyy(jat)  ; dz23(i1) = zzz(kat)-zzz(jat)
  endif
 enddo

 if (i1 > 0) then
 call periodic_images(dx12(1:i1),dy12(1:i1),dz12(1:i1))
 call periodic_images(dx23(1:i1),dy23(1:i1),dz23(1:i1))
 r12_sq(1:i1) = dx12(1:i1)* dx12(1:i1)+dy12(1:i1)*dy12(1:i1)+dz12(1:i1)*dz12(1:i1)
 r23_sq(1:i1) = dx23(1:i1)* dx23(1:i1)+dy23(1:i1)*dy23(1:i1)+dz23(1:i1)*dz23(1:i1)

 i1 = 0
 
 
  do iangle = 1+rank, Nangles, nprocs
  iat = list_angles(1,iangle)  ;  jat = list_angles(2,iangle)  ;  kat = list_angles(3,iangle)
  if (iat==iwhich.or.jat==iwhich.or.kat==iwhich) then
  itype = list_angles(0,iangle)
  i1 = i1 + 1
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
  endif !if (iat==iwhich.or.jat==iwhich.or.kat==iwhich)
 enddo ! iangle = 1, Nangles
 endif ! i1>0
 d_en_angle = local_energy

 deallocate(r12_sq,r23_sq)
 deallocate(dx23,dy23,dz23)
 deallocate(dx12,dy12,dz12)

end subroutine angle_ENERGY_MCmove_1atom


!------------------
!-----------------


!----DIHEDRALS ---
!-----------------

subroutine dihedral_forces
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
 call initialize
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
         ff = ff + dble(i)*b*(dsin(dble(i)*angle ) ) 
      enddo
      ff = ff * sin_inv * i_sqrt_t2_u2
    case(2) ! improper

      angle = angle-phase
      b = prm_dihedral_types(1,itype)
!      angle = angle-Pi2*(dble(int(2.0d0*(angle*i_pi2)))-dble(int(angle*i_pi2))     )   ! recenter it
      En = 0.5d0*b*angle*angle
      ff = b * angle * i_sqrt_t2_u2 * sin_inv

    case default
      print*, ' DIHEDRAL CASE NOT DEFINED IN dihedral_forces',iStyle
      STOP
  end select   
!if (istyle==2)write(14,*) itemp,ii,jj,kk,ll

 fct1 = ut*i_t2   ;   fct2 = ut*i_u2
 fx1 = ff*((-uy*z23+uz*y23) + fct1*( ty*z23-tz*y23))
 fy1 = ff*(( ux*z23-uz*x23) - fct1*( tx*z23-tz*x23))
 fz1 = ff*((-ux*y23+uy*x23) + fct1*( tx*y23-ty*x23))

 fx2 = ff*((-ty*z34+tz*y34) + fct2*( uy*z34-uz*y34))
 fy2 = ff*(( tx*z34-tz*x34) - fct2*( ux*z34-uz*x34))
 fz2 = ff*((-tx*y34+ty*x34) + fct2*( ux*y34-uy*x34))

 fx3 = ff*((-uy*z12+uz*y12) + fct1*( ty*z12-tz*y12))
 fy3 = ff*(( ux*z12-uz*x12) - fct1*( tx*z12-tz*x12))
 fz3 = ff*((-ux*y12+uy*x12) + fct1*( tx*y12-ty*x12))

 fx4 = ff*((-ty*z23+tz*y23) + fct2*( uy*z23-uz*y23))
 fy4 = ff*(( tx*z23-tz*x23) - fct2*( ux*z23-uz*x23))
 fz4 = ff*((-tx*y23+ty*x23) + fct2*( ux*y23-uy*x23))

 fxx(ii) = fxx(ii) + fx1  ;  fyy(ii) = fyy(ii) + fy1  ;  fzz(ii) = fzz(ii) + fz1
 fxx(ll) = fxx(ll) + fx4  ;  fyy(ll) = fyy(ll) + fy4  ;  fzz(ll) = fzz(ll) + fz4
 fxx(jj) = fxx(jj) - fx1 + fx2 - fx3 ;  fyy(jj) = fyy(jj) - fy1 + fy2 - fy3 ;  fzz(jj) = fzz(jj) - fz1 + fz2 - fz3
 fxx(kk) = fxx(kk) - fx2 + fx3 - fx4 ;  fyy(kk) = fyy(kk) - fy2 + fy3 - fy4 ;  fzz(kk) = fzz(kk) - fz2 + fz3 - fz4

 fx23 = fx2-fx3  ; fy23 = fy2-fy3  ;  fz23 = fz2-fz3
 sxx = x12*fx1+x23*fx23-x34*fx4
 sxy = y12*fx1+y23*fx23-y34*fx4
 sxz = z12*fx1+z23*fx23-z34*fx4
 syy = y12*fy1+y23*fy23-y34*fy4
 syz = y12*fz1+y23*fz23-y34*fz4
 szz = z12*fz1+z23*fz23-z34*fz4

 local_energy = local_energy + En
 
   lst_xx = lst_xx + sxx
   lst_xy = lst_xy + sxy
   lst_xz = lst_xz + sxz
   lst_yy = lst_yy + syy
   lst_zz = lst_zz + szz
   lst_yz = lst_yz + syz


! if (l_need_2nd_profile) then
!  call increment_profiles(dih_profile(idihedral), En,sxx,sxy,sxz,syy,syz,szz,fxx,fxy,fxz)
! endif


 enddo 

 en_dih = local_energy
 stress_dih(1) = lst_xx
 stress_dih(2) = lst_yy
 stress_dih(3) = lst_zz
 stress_dih(4) = (lst_xx+lst_yy+lst_zz)/3.0d0
 stress_dih(5) = lst_xy
 stress_dih(6) = lst_xz
 stress_dih(7) = lst_yz
 stress_dih(8) = lst_xy
 stress_dih(9) = lst_xz
 stress_dih(10)= lst_yz
 stress = stress + stress_dih
!print*, 'en dih = ',en_dih/418.4d0
!do i = 1, ubound(fxx,dim=1)
! write(14,*) i,fxx(i)/418.4d0,fyy(i)/418.4d0,fzz(i)/418.4d0
!enddo
!print*, 'sum forces=',sum(fxx),sum(fyy),sum(fzz)
!print*,'stress dihedrals=',stress_dih/418.4d0
 deallocate(r12_sq,r23_sq,r34_sq)
 deallocate(dx34,dy34,dz34)
 deallocate(dx23,dy23,dz23)
 deallocate(dx12,dy12,dz12)

 contains
  subroutine initialize
  local_energy = 0.0d0
   lst_xx = 0.0d0
   lst_xy = 0.0d0
   lst_xz = 0.0d0
   lst_yy = 0.0d0
   lst_zz = 0.0d0
   lst_yz = 0.0d0

!   if (l_need_2nd_profile) then
!    do i = 1+rank, ,nprocs
!      call zero_atom_profile(dihedral_profile(i))
!    enddo
!    endif
 end subroutine initialize
 
end subroutine dihedral_forces

!--------------------------------------

subroutine dihedral_ENERGY
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

 end subroutine dihedral_ENERGY 

!--------------------------------------

subroutine dihedral_ENERGY_MCmove_1atom(iwhich)
use connectivity_type_data, only : dihedral_types, prm_dihedral_types, Nfolds_dihedral_types
use connectivity_ALL_data, only : list_dihedrals
use ALL_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz
use boundaries, only : periodic_images
use sizes_data, only : Ndihedrals
use d_energies_data, only : d_en_dih
use paralel_env_data
use stresses_data, only : stress_dih,stress
use math_constants, only : Pi2
use paralel_env_data
implicit none
integer, intent(IN) :: iwhich
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
  i = list_dihedrals(1,idihedral) ; j = list_dihedrals(2,idihedral)
  k = list_dihedrals(3,idihedral) ; l = list_dihedrals(4,idihedral)
  if (i==iwhich.or.j==iwhich.or.k==iwhich.or.l==iwhich) then
  i1 = i1 + 1
  dx12(i1)=xxx(i)-xxx(j)  ;  dy12(i1)=yyy(i)-yyy(j)  ;  dz12(i1)=zzz(i)-zzz(j)
  dx23(i1)=xxx(j)-xxx(k)  ;  dy23(i1)=yyy(j)-yyy(k)  ;  dz23(i1)=zzz(j)-zzz(k)
  dx34(i1)=xxx(k)-xxx(l)  ;  dy34(i1)=yyy(k)-yyy(l)  ;  dz34(i1)=zzz(k)-zzz(l)
  endif
 enddo

 if (i1 > 0) then
 call periodic_images(dx12(1:i1),dy12(1:i1),dz12(1:i1))
 call periodic_images(dx23(1:i1),dy23(1:i1),dz23(1:i1))
 call periodic_images(dx34(1:i1),dy34(1:i1),dz34(1:i1))
 r12_sq(1:i1) = dx12(1:i1)* dx12(1:i1)+dy12(1:i1)*dy12(1:i1)+dz12(1:i1)*dz12(1:i1)
 r23_sq(1:i1) = dx23(1:i1)* dx23(1:i1)+dy23(1:i1)*dy23(1:i1)+dz23(1:i1)*dz23(1:i1)
 r34_sq(1:i1) = dx34(1:i1)* dx34(1:i1)+dy34(1:i1)*dy34(1:i1)+dz34(1:i1)*dz34(1:i1)
itemp=0

 i1 = 0
 do idihedral = 1+rank, Ndihedrals , nprocs
  ii = list_dihedrals(1,idihedral) ; jj = list_dihedrals(2,idihedral)
  kk = list_dihedrals(3,idihedral) ; ll = list_dihedrals(4,idihedral)
  if (ii==iwhich.or.jj==iwhich.or.kk==iwhich.or.ll==iwhich) then
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
 endif ! 
 enddo
 endif ! i1 > 0
 d_en_dih = local_energy
 deallocate(r12_sq,r23_sq,r34_sq)
 deallocate(dx34,dy34,dz34)
 deallocate(dx23,dy23,dz23)
 deallocate(dx12,dy12,dz12)

 end subroutine dihedral_ENERGY_MCmove_1atom

!-----------------
!------------------
!----------------

!---DEFORMS-----
subroutine out_of_plane_deforms
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
      ff = b * angle * i_sqrt_t2_u2 / cos_dihedral
      fct1 = ut * i_t2    ;   fct2 = ut * i_u2

        fx4 =  (tx - (ux*fct2))*ff
        fy4 =  (ty - (uy*fct2))*ff 
        fz4 =  (tz - (uz*fct2))*ff

        fx1 = ff*(fct1*(tz*y23 - ty*z23) -  (uz*y23 - uy*z23))
        fy1 = ff*(fct1*(tx*z23 - tz*x23) -  (ux*z23 - uz*x23))
        fz1 = ff*(fct1*(ty*x23 - tx*y23) -  (uy*x23 - ux*y23))

        fx3 = ff*(fct1*(tz*y12 - ty*z12) -  (uz*y12 - uy*z12))
        fy3 = ff*(fct1*(tx*z12 - tz*x12) -  (ux*z12 - uz*x12))
        fz3 = ff*(fct1*(ty*x12 - tx*y12) -  (uy*x12 - ux*y12))

        fx2 = - (fx1 +fx3 +fx4 )
        fy2 = - (fy1 +fy3 +fy4 )
        fz2 = - (fz1 +fz3 +fz4 )

        fxx(ii) = fxx(ii) - fx1 ; fyy(ii) = fyy(ii) - fy1 ; fzz(ii) = fzz(ii) - fz1 ;
        fxx(jj) = fxx(jj) - fx2 ; fyy(jj) = fyy(jj) - fy2 ; fzz(jj) = fzz(jj) - fz2 ;
        fxx(kk) = fxx(kk) - fx3 ; fyy(kk) = fyy(kk) - fy3 ; fzz(kk) = fzz(kk) - fz3 ;
        fxx(ll) = fxx(ll) - fx4 ; fyy(ll) = fyy(ll) - fy4 ; fzz(ll) = fzz(ll) - fz4 ;


          x13=x12 + x23 ; y13 = y12 +y23; z13 = z12 +z23
          x14=x12 + x24 ; y14 = y12 +y24; z14 = z12 +z24
          pos1_xx = xx(ii) ;              pos1_yy = yy(ii) ;             pos1_zz = zz(ii)
          pos2_xx = pos1_xx + x12 ;       pos2_yy = pos1_yy + y12 ;      pos2_zz = pos1_zz + z12 ; 
          pos3_xx = pos1_xx + x13 ;       pos3_yy = pos1_yy + y13 ;      pos3_zz = pos1_zz + z13 ; 
          pos4_xx = pos1_xx + x14 ;       pos4_yy = pos1_yy + y14 ;      pos4_zz = pos1_zz + z14 ; 

          sxx = fx1*pos1_xx + fx2*pos2_xx + fx3*pos3_xx + fx4*pos4_xx
          sxy = fx1*pos1_yy + fx2*pos2_yy + fx3*pos3_yy + fx4*pos4_yy
          sxz = fx1*pos1_zz + fx2*pos2_zz + fx3*pos3_zz + fx4*pos4_zz
          
          syx = fy1*pos1_xx + fy2*pos2_xx + fy3*pos3_xx + fy4*pos4_xx
          syy = fy1*pos1_yy + fy2*pos2_yy + fy3*pos3_yy + fy4*pos4_yy
          syz = fy1*pos1_zz + fy2*pos2_zz + fy3*pos3_zz + fy4*pos4_zz
          
          szx = fz1*pos1_xx + fz2*pos2_xx + fz3*pos3_xx + fz4*pos4_xx
          szy = fz1*pos1_yy + fz2*pos2_yy + fz3*pos3_yy + fz4*pos4_yy
          szz = fz1*pos1_zz + fz2*pos2_zz + fz3*pos3_zz + fz4*pos4_zz
          
          lst_xx = lst_xx + sxx
          lst_xy = lst_xy + sxy
          lst_xz = lst_xz + sxz

          lst_yx = lst_yx + syx
          lst_yy = lst_yy + syy
          lst_yz = lst_yz + syz 
          
          lst_zx = lst_zx + szx
          lst_zy = lst_zy + szy
          lst_zz = lst_zz + szz    

!write(14,*)i_deforms,ii,jj,kk,ll
!write(14,*)i_deforms,fx1/418.4d0,fy1/418.4d0,fz1/418.4d0
!write(14,*)i_deforms,fx2/418.4d0,fy2/418.4d0,fz2/418.4d0
!write(14,*)i_deforms,fx3/418.4d0,fy3/418.4d0,fz3/418.4d0
!write(14,*)i_deforms,fx4/418.4d0,fy4/418.4d0,fz4/418.4d0

    end select



   local_energy = local_energy + En
  enddo


   en_deform = local_energy

   stress_deform(1) = lst_xx
   stress_deform(2) = lst_yy
   stress_deform(3) = lst_zz
   stress_deform(4) = (lst_xx+lst_yy+lst_zz)/3.0d0
   stress_deform(5) = lst_xy
   stress_deform(6) = lst_xz
   stress_deform(7) = lst_yz
   stress_deform(8) = lst_yx
   stress_deform(9) = lst_zx
   stress_deform(10)= lst_zy
   stress = stress + stress_deform

!print*, 'en_deform=',en_deform/418.4d0
!print*, 'sum_forces=',sum(fxx),sum(fyy),sum(fzz)
!print*, 'stress deform = ',stress_deform/418.4d0
!do i = 1, ubound(fxx,dim=1)
!write(14,*)i,fxx(i)/418.4d0,fyy(i)/418.0d0,fzz(i)/418.0d0
!enddo


 deallocate(r12_sq,r23_sq,r24_sq)
 deallocate(dx24,dy24,dz24)
 deallocate(dx23,dy23,dz23)
 deallocate(dx12,dy12,dz12)

 CONTAINS
  subroutine initialize
  local_energy = 0.0d0
   lst_xx = 0.0d0
   lst_xy = 0.0d0
   lst_xz = 0.0d0
   lst_yy = 0.0d0
   lst_zz = 0.0d0
   lst_yz = 0.0d0
  lst_yx=0.0d0
  lst_zx=0.0d0
  lst_zy=0.0d0

!   if (l_need_2nd_profile) then
!    do i = 1+rank, ,nprocs
!      call zero_atom_profile(dihedral_profile(i))
!    enddo
!    endif
 end subroutine initialize

end subroutine out_of_plane_deforms

!---------------------------

 subroutine out_of_plane_deforms_ENERGY
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
 local_energy=0.0d0
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

 deallocate(dx12,dy12,dz12)
 deallocate(dx23,dy23,dz23)
 deallocate(dx24,dy24,dz24)
 deallocate(r12_sq,r23_sq,r24_sq)

end subroutine out_of_plane_deforms_ENERGY


!---------------------------
!--------------------------
 subroutine out_of_plane_deforms_ENERGY_MCmove_1atom(iwhich)
use connectivity_type_data, only : deform_types, prm_deform_types
use connectivity_ALL_data, only : list_deforms
use ALL_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz,xx,yy,zz
use boundaries, only : periodic_images
use sizes_data, only : Ndeforms
use d_energies_data, only : d_en_deform
use paralel_env_data
use stresses_data, only : stress_deform,stress
use math_constants, only : Pi2
use paralel_env_data
implicit none
integer, intent(IN) :: iwhich
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
 local_energy=0.0d0
 i1 = 0
 do i_deforms = 1+rank, Ndeforms, nprocs
  i = list_deforms(1,i_deforms) ; j = list_deforms(2,i_deforms)
  k = list_deforms(3,i_deforms) ; l = list_deforms(4,i_deforms)
  if (i==iwhich.or.j==iwhich.or.k==iwhich.or.l==iwhich) then
  i1 = i1 + 1
  dx12(i1)=xxx(i)-xxx(j)  ;  dy12(i1)=yyy(i)-yyy(j)  ;  dz12(i1)=zzz(i)-zzz(j)
  dx23(i1)=xxx(j)-xxx(k)  ;  dy23(i1)=yyy(j)-yyy(k)  ;  dz23(i1)=zzz(j)-zzz(k)
  dx24(i1)=xxx(j)-xxx(l)  ;  dy24(i1)=yyy(j)-yyy(l)  ;  dz24(i1)=zzz(j)-zzz(l)
  endif
 enddo

 if (i1 > 0 ) then
 call periodic_images(dx12(1:i1),dy12(1:i1),dz12(1:i1))
 call periodic_images(dx23(1:i1),dy23(1:i1),dz23(1:i1))
 call periodic_images(dx24(1:i1),dy24(1:i1),dz24(1:i1))
 r12_sq(1:i1) = dx12(1:i1)* dx12(1:i1)+dy12(1:i1)*dy12(1:i1)+dz12(1:i1)*dz12(1:i1)
 r23_sq(1:i1) = dx23(1:i1)* dx23(1:i1)+dy23(1:i1)*dy23(1:i1)+dz23(1:i1)*dz23(1:i1)
 r24_sq(1:i1) = dx24(1:i1)* dx24(1:i1)+dy24(1:i1)*dy24(1:i1)+dz24(1:i1)*dz24(1:i1)

itemp=0

  i1 = 0
  do i_deforms = 1+rank, Ndeforms, nprocs
    ii = list_deforms(1,i_deforms) ; jj = list_deforms(2,i_deforms)
    kk = list_deforms(3,i_deforms) ; ll = list_deforms(4,i_deforms)
    if (ii==iwhich.or.jj==iwhich.or.kk==iwhich.or.ll==iwhich) then
    i1 = i1 + 1
    x12 = dx12(i1) ; y12 = dy12(i1) ; z12 = dz12(i1)
    x23 = dx23(i1) ; y23 = dy23(i1) ; z23 = dz23(i1)
    x24 = dx24(i1) ; y24 = dy24(i1) ; z24 = dz24(i1) ! extract them in scalars for faster memory acces


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
   endif
  enddo
  endif
   d_en_deform = local_energy
 deallocate(dx12,dy12,dz12)
 deallocate(dx23,dy23,dz23)
 deallocate(dx24,dy24,dz24)
 deallocate(r12_sq,r23_sq,r24_sq)

end subroutine out_of_plane_deforms_ENERGY_MCmove_1atom
!----------------------------------
!--------------------------------
!- done with dihedrals

subroutine increment_profiles(V,A0,A1,A2,A3,A4,A5,A6,F1,F2,F3)
use types_module, only : atom_profile_type
 type(atom_profile_type) V
 real(8), intent(IN) :: A0,A1,A2,A3,A4,A5,A6, F1,F2,F3
 V%pot = V%pot + A0
 V%sxx = V%sxx + A1
 V%sxy = V%sxy + A2
 V%sxz = V%sxz + A3
 V%syy = V%syy + A4
 V%syz = V%syz + A5
 V%szz = V%szz + A6
! V%fxx = V%fxx + F1
! V%fyy = V%fyy + F2
! V%fzz = V%fzz + F3
! V%fff = V%fff + dsqrt(f1*f1+f2*f2+f3*f3) 
end subroutine increment_profiles

subroutine zero_atom_profile(V)
use types_module, only : atom_profile_type
type(atom_profile_type) V
 V%pot = 0.0d0
 V%sxx = 0.0d0
 V%sxy = 0.0d0 
 V%sxz = 0.0d0
 V%syy = 0.0d0
 V%syz = 0.0d0
 V%szz = 0.0d0
! V%fxx = 0.0d0
! V%fyy = 0.0d0
! V%fzz = 0.0d0
! V%fff = 0.0d0
end subroutine zero_atom_profile
end module intramolecular_forces
