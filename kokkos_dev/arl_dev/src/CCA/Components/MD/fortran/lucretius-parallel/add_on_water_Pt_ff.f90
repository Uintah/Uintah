
 module water_Pt_ff_module
! add the extra force_field terms for water - Pt potential
implicit none

public :: driver_water_surface_ff
public :: driver_water_surface_ff_ENERGY
private :: very_first_pass
private :: update_water_M_pairs
private :: add_force_field
private :: add_force_field_ENERGY

type,private :: which_atom_type
 integer :: Ow
 integer :: Surf   
end type which_atom_type
type(which_atom_type),allocatable,private :: which_atom(:)

logical, private, save :: flag_water_surface_CTRL = .false.
integer, private :: MAX_WATER_SURF_PAIRS  = 200 
real(8), private :: densvar_water_surf_pairs ! a parameter to increase/decrease size of pair vectors.
real(8), private :: DD_ff = 1435.0d0  ! kJ/mol A^3
real(8), private :: scut_ff = 3.2d0 !3.2d0 ! Amstrom
real(8), private :: scut2_ff 
real(8), private :: BB_ff = 0.6d0 ! Amstrom
! above 3 add the extra-water-dipole ff 
! Ud = -DD_ff*dexp(BB_ff/(r-scut_ff))*dexp(-8*[(cos(fi)-1)/4]^4)
real(8), private :: EE_ff = 41.0d0 !  jK/mol A^3   ! 3-body
real(8), private :: FF_ff = 13.3d0                 ! 3-body
real(8), private :: beta_ff = 10.0d0               ! 3-body
!  above 3 add the 3 body potenital:
! U3 = EE_ff*dexp(BB_ff/(r-scut_ff))/r^beta_ff * exp(FF_ff*cos^2(theta_ijk/2.0))
real(8), private :: RANGE_RADIAL_SURF_SURF = 2.6d0  ! range token from the surface characteristics
real(8), private :: RANGE_ZZ_SURF_SURF = 1.0d0   ! range toten from the surface characteristics
integer, private :: MX_Next_SURF = 20             ! again to be seen from the surface characteristis

integer, private :: N_water, N_Surface, N_pairs
real(8), private :: amplify_a_bit = 1.20 ! how much to amplify the size of MAX_WATER_SURF_PAIRS

logical, allocatable, private :: is_surface(:), is_water(:)
integer, allocatable, private :: Size_Next_Surf(:), next_Surf(:,:)

CONTAINS

subroutine driver_water_surface_ff
use non_bonded_lists_data, only : l_update_VERLET_LIST
use integrate_data, only : integration_step
 logical, save :: l_very_first_pass = .true.
 if (l_very_first_pass) then
    call very_first_pass
    l_very_first_pass = .false.
 endif
 if (.not.flag_water_surface_CTRL) RETURN
 if (l_update_VERLET_LIST) call  update_water_M_pairs
 call add_force_field
end subroutine driver_water_surface_ff

subroutine driver_water_surface_ff_ENERGY
use non_bonded_lists_data, only : l_update_VERLET_LIST
use integrate_data, only : integration_step
! must be called after driver_water_surface_ff
 if (.not.flag_water_surface_CTRL) RETURN
 call add_force_field_ENERGY
end subroutine driver_water_surface_ff_ENERGY


subroutine very_first_pass
use all_atoms_data, only : xxx,yyy,zzz,i_type_atom, Natoms
use ALL_mols_data, only : i_type_molecule, start_group,end_group,Nmols
use mol_type_data, only : mol_type_name
use atom_type_data, only : atom_type_name
use boundaries, only : periodic_images
use sim_cel_data, only : sim_cel, Volume
use cut_off_data, only : displacement
use physical_constants, only : LJ_epsilon_convert
implicit none
integer i,j,k,imol,i1,LS,LS1,iat,jat
real(8) dx(100),dy(100),dz(100),dr_sq(100) ! some working space
logical i_S,j_S
real(8) dN
integer estimated_N

allocate(is_water(Natoms),is_surface(Natoms)) ; is_water=.false.;is_surface=.false.
allocate(Size_Next_Surf(Natoms),Next_Surf(Natoms, MX_Next_SURF)) ; Size_next_surf=0 ; Next_Surf=0

DD_ff = DD_ff * LJ_epsilon_convert ! convert it in internal units
EE_ff = EE_ff * LJ_epsilon_convert

scut2_ff = scut_ff*scut_ff
i1 = 0
do i = 1, Nmols
 imol = i_type_molecule(i)
 LS = len('water')
 if (len(trim(mol_type_name(imol))) >= LS ) then
 if (mol_type_name(imol)(1:LS)=='water'.or.mol_type_name(imol)(1:LS)=='WATER'.or.mol_type_name(imol)(1:LS)=='Water') then
   LS1 = len('ows')
   do j = start_group(i),end_group(i) 
   iat = i_type_atom(j)
   if (len(trim(atom_type_name(iat)))>=LS1) then
   if (atom_type_name(iat)(1:LS1)=='Ows'.or.atom_type_name(iat)(1:LS1)=='OWS'.or.atom_type_name(iat)(1:LS1)=='ows') then
    i1 = i1 + 1 
    is_water(j) = .true.
   endif
   endif
   enddo
 endif
 endif
enddo
N_water = i1
i1 = 0 
LS = len('surface')
do i = 1, Nmols
 imol = i_type_molecule(i)
 if (len(trim(mol_type_name(imol))) >= LS ) then
 if (mol_type_name(imol)(1:LS)=='Surface'.or.mol_type_name(imol)(1:LS)=='surface'.or.mol_type_name(imol)(1:LS)=='SURFACE') then
 do j = start_group(i),end_group(i)
    i1 = i1  + 1
    is_surface(j) = .true.
 enddo
 endif
 endif
enddo
N_Surface = i1

if (N_water > 0 .and. N_surface > 0) flag_water_surface_CTRL = .true.
print*, 'Nwater Nsurface = ',N_water,N_Surface, flag_water_surface_CTRL

if (.not.flag_water_surface_CTRL) then
 deallocate(is_water,is_surface)
 deallocate(Size_Next_Surf,Next_Surf)
 RETURN ! DO NOTHING 
endif

do i = 1, Natoms
  if (is_surface(i)) then
    do j = 1, Natoms
      if (is_surface(j)) then
      if (i /= j) then
         dx(1) = xxx(i)-xxx(j)
         dy(1) = yyy(i)-yyy(j)
         dz(1) = zzz(i)-zzz(j)
         call periodic_images(dx(1:1),dy(1:1),dz(1:1))
         dr_sq(1:1) = dx(1:1)**2+dy(1:1)**2+dz(1:1)**2
         if (dr_sq(1) <= (RANGE_RADIAL_SURF_SURF**2))  then
         if (dz(1) <= RANGE_ZZ_SURF_SURF) then 
            Size_next_surf(i) = Size_Next_Surf(i) + 1
            if (Size_next_surf(i) > MX_Next_SURF) then
             print*, 'ERROR in water_Pt_ff_module%very_first_pass; Size_next_surf(i) > MX_Next_SURF'
             print*, 'Increase the number of surface neighbours as necesary'
             STOP
            endif
            Next_Surf(i,Size_Next_Surf(i)) = j
         endif
         endif
      endif
      endif
    enddo
  endif
enddo

! Get a better estimate of MAX_WATER_SURF_PAIRS

 dN = 4.0d0/3.0d0*3.14d0*( scut_ff + displacement)**3 * amplify_a_bit
 dN = dN * Natoms/Volume   * 2.0d0 * 2.0d0
 estimated_N = NINT(dN) / 2.0d0
 MAX_WATER_SURF_PAIRS = estimated_N*estimated_N +1000
print*, 'estimated_N MAX_WATER_SURF_PAIRS=',estimated_N,MAX_WATER_SURF_PAIRS 
allocate(which_atom(MAX_WATER_SURF_PAIRS));which_atom%Ow=0; which_atom%Surf=0
call update_water_M_pairs
end subroutine very_first_pass


subroutine update_water_M_pairs
use cut_off_data, only : displacement
use all_atoms_data, only : Natoms, xxx,yyy,zzz
use non_bonded_lists_data, only : list_nonbonded,size_list_nonbonded,l_update_VERLET_LIST
use boundaries, only : periodic_images
implicit none
real(8) local_cut, local_cut_sq,dx(10),dy(10),dz(10)
integer i,j,k
integer  i_pair
logical i_water,j_water,i_S,j_S
logical l_1,l_2

local_cut = scut_ff + displacement
local_cut_sq = local_cut*local_cut
i_pair = 0
do i = 1, Natoms - 1
i_S = is_surface(i)
i_water = is_water(i)
if (i_S.or.i_water) then
 do k = 1, size_list_nonbonded(i)
  j = list_nonbonded(i,k)
  j_S = is_surface(j)
  j_water = is_water(j)
  if (j_S.or.j_water) then
  l_1 = i_S.and.j_water
  l_2 = j_S.and.i_water
  if (l_1) then 
  dx(1) = xxx(i) - xxx(j)
  dy(1) = yyy(i) - yyy(j)
  dz(1) = zzz(i) - zzz(j)
  call periodic_images(dx(1:1),dy(1:1),dz(1:1))
  if (dx(1)**2+dy(1)**2+dz(1)**2 <= local_cut_sq) then 
    i_pair = i_pair + 1
    if (i_pair > MAX_WATER_SURF_PAIRS) then
      print*, 'ERROR in update_water_M_pairs; i_pair > MAX_WATER_SURF_PAIRS',i_pair,MAX_WATER_SURF_PAIRS
      print*, 'INCREASE MAX_WATER_S_PAIRS and restart'
      STOP 
    endif
    which_atom(i_pair)%Ow = j
    which_atom(i_pair)%Surf = i
  endif
  else if (l_2) then
  dx(1) = xxx(i) - xxx(j)
  dy(1) = yyy(i) - yyy(j)
  dz(1) = zzz(i) - zzz(j)
  call periodic_images(dx(1:1),dy(1:1),dz(1:1))
  if (dx(1)**2+dy(1)**2+dz(1)**2 <= local_cut_sq) then
    i_pair = i_pair + 1
    if (i_pair > MAX_WATER_SURF_PAIRS) then
      print*, 'ERROR in update_water_M_pairs; i_pair > MAX_WATER_S_PAIRS',i_pair,MAX_WATER_SURF_PAIRS
      print*, 'INCREASE MAX_WATER_S_PAIRS and restart'
      STOP
    endif
    which_atom(i_pair)%Ow = i
    which_atom(i_pair)%Surf = j
  endif  
  endif
 endif
 enddo 
endif
enddo

N_pairs = i_pair
end subroutine update_water_M_pairs

subroutine add_force_field
use mol_utils, only : get_all_mol_dipoles
use boundaries, only : periodic_images
use all_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz, all_charges,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,Natoms,&
                           atom_in_which_molecule
use all_mols_data, only : Nmols, mol_dipole, start_group,end_group
use energies_data
implicit none
integer i,j,k,i_O,i_S,i_mol,i_pair,i1,kk,Next_surf_atom
real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:),dx_jk(:),dy_jk(:),dz_jk(:),dr_sq_jk(:)
real(8) x,y,z, qi,qj,qk,dj_xx,dj_yy,dj_zz,r,r2,i_r,i_r2,i_r3,f2,f3,f4,f23,f24,wd,r_wr,i_r_wd,wd2,i_wd2
real(8) ff1_xx,ff1_yy,ff1_zz,ff2_xx,ff2_yy,ff2_zz,ff_xx,ff_yy,ff_zz
real(8) coef5,cos_fi, arg_f3, En, local_energy, i_r_ijB, CC1,CC2, i_C2,i_C2_f2
real(8) f23_cos,fct_cos_fi, r_wd
real(8) x_jk,y_jk,z_jk
real(8) r_jk,r2_jk, i_r_r, cos_ijk,ps_jk,arj_ijk, arg_jk, ps
real(8) f_xx,f_yy,f_zz,ff1_xx_i,ff1_yy_i,ff1_zz_i,ff1_xx_j,ff1_yy_j,ff1_zz_j
real(8) ff2_xx_i,ff2_yy_i,ff2_zz_i,ff2_xx_j,ff2_yy_j,ff2_zz_j
real(8) ff1_xx_k,ff1_yy_k,ff1_zz_k,ff2_xx_k,ff2_yy_k,ff2_zz_k
real(8) ff3_xx_i,ff3_yy_i,ff3_zz_i,ff3_xx_j,ff3_yy_j,ff3_zz_j,ff3_xx_k,ff3_yy_k,ff3_zz_k
real(8) fi3,arg_fi3, A, A1, B, beta_i_r2, i_r_jk_2, i_r_jk,pr_i_r2,i_C2_r,pr, i_rij_at_beta
real(8) CC,t2_x,t2_y,t2_z, s_fk_x,s_fk_y,s_fk_z
real(8) local_energy_attract, local_energy_repulsion
real(8) ff1_xx_S,ff1_yy_S,ff1_zz_S,ff1_xx_O,ff1_yy_O,ff1_zz_O
real(8) ff2_xx_S,ff2_yy_S,ff2_zz_S,ff2_xx_O,ff2_yy_O,ff2_zz_O
real(8) ff3_xx_S,ff3_yy_S,ff3_zz_S,ff3_xx_O,ff3_yy_O,ff3_zz_O

allocate(dx(N_pairs),dy(N_pairs),dz(N_pairs),dr_sq(N_pairs))
allocate(dx_jk(MX_Next_SURF),dy_jk(MX_Next_SURF),dz_jk(MX_Next_SURF),dr_sq_jk(MX_Next_SURF))
local_energy_attract = 0.0d0
local_energy_repulsion = 0.0d0
  call get_all_mol_dipoles
  do i_pair = 1, N_pairs
    i_O = which_atom(i_pair)%Ow
    i_S = which_atom(i_pair)%Surf
    dx(i_pair) = xxx(i_O) - xxx(i_S)
    dy(i_pair) = yyy(i_O) - yyy(i_S)
    dz(i_pair) = zzz(i_O) - zzz(I_S)
  enddo
  call periodic_images(dx(1:N_pairs),dy(1:N_pairs),dz(1:N_pairs))
  dr_sq(1:N_pairs) = dx(1:N_pairs)**2+dy(1:N_pairs)**2+dz(1:N_pairs)**2

  do i_pair = 1, N_pairs
    i_O = which_atom(i_pair)%Ow
    i_S = which_atom(i_pair)%Surf
    r2 = dr_sq(i_pair)
    if (r2 <= scut2_ff) then
      r = dsqrt(r2) ; i_r = 1.0d0/r; i_r2 = i_r*i_r ; i_r3 = i_r2*i_r
      x = dx(i_pair) ; y = dy(i_pair) ; z = dz(i_pair)
      f2 = dexp(BB_ff/(r-scut_ff))
      i_mol = atom_in_which_molecule(i_O)
      dj_xx =  mol_dipole(i_mol,1); dj_yy =  mol_dipole(i_mol,2); dj_zz =  mol_dipole(i_mol,3)
      qj = all_charges(i_O)
      pr = dj_xx*x + dj_yy*y + dj_zz*z
      wd2 = dj_xx*dj_xx + dj_yy*dj_yy + dj_zz*dj_zz 
      wd = dsqrt(wd2)  ; i_wd2 = 1.0d0/wd2
      r_wd= r*wd
      i_r_wd = 1.0d0/r_wd
      cos_fi = pr*i_r_wd
      arg_f3 = (cos_fi-1.0d0)*0.25d0
      f3 = dexp(-8.0d0*(arg_f3*arg_f3)*(arg_f3*arg_f3))
      f23 = f2*f3
      En = - DD_ff*f23 * i_r3
      local_energy_attract = local_energy_attract + En
      i_C2 = 1.0d0/((r-scut_ff)*(r-scut_ff))
      i_C2_r = i_C2 * i_r
      CC=(BB_ff*i_C2_r) * En
      ff1_xx_O = CC*x
      ff1_yy_O = CC*y
      ff1_zz_O = CC*z
      CC2 = (3.0d0*En*i_r2)
      ff3_xx_O = CC2*x
      ff3_yy_O = CC2*y
      ff3_zz_O = CC2*z
      pr_i_r2 = pr*i_r2

      CC = -En*8.0d0*arg_f3*arg_f3*arg_f3
      ff2_xx_S = CC*(i_r_wd*(-dj_xx+pr_i_r2*x )) 
      ff2_yy_S = CC*(i_r_wd*(-dj_yy+pr_i_r2*y ))
      ff2_zz_S = CC*(i_r_wd*(-dj_zz+pr_i_r2*z ))

      A = (i_r_wd)*CC ; B = (cos_fi/wd2)*CC
      f_xx = A*x-B*dj_xx
      f_yy = A*y-B*dj_yy
      f_zz = A*z-B*dj_zz
      ff2_xx_O = -ff2_xx_S - f_xx*qj 
      ff2_yy_O = -ff2_yy_S - f_yy*qj
      ff2_zz_O = -ff2_zz_S - f_zz*qj
      do k = start_group(i_mol),end_group(i_mol)
      if (k /= i_O) then
!print*,'----- r = ',r
      qk = all_charges(k)
         fxx(k) = fxx(k) - f_xx*qk
         fyy(k) = fyy(k) - f_yy*qk
         fzz(k) = fzz(k) - f_zz*qk
!print*, 'fH=',f_xx*qk,f_yy*qk,f_zz*qk
      endif
      enddo

      fxx(i_O) = fxx(i_O) + ff1_xx_O + ff3_xx_O + ff2_xx_O
      fyy(i_O) = fyy(i_O) + ff1_yy_O + ff3_yy_O + ff2_yy_O
      fzz(i_O) = fzz(i_O) + ff1_zz_O + ff3_zz_O + ff2_zz_O

      fxx(i_S) = fxx(i_S) - ff1_xx_O - ff3_xx_O + ff2_xx_S
      fyy(i_S) = fyy(i_S) - ff1_yy_O - ff3_yy_O + ff2_yy_S
      fzz(i_S) = fzz(i_S) - ff1_zz_O - ff3_zz_O + ff2_zz_S
! Now add the 3rd body term
      i1 = 0
      i_rij_at_beta = 1.0d0/(r**beta_ff)
      CC = EE_ff*i_rij_at_beta*f2  
      beta_i_r2 = beta_ff * i_r2
      Next_surf_atom =  Size_Next_Surf(i_S)
      do k = 1, Next_surf_atom
        kk = Next_Surf(i_S,k)
        i1  = i1 + 1
        dx_jk(k) = xxx(i_S) - xxx(kk)
        dy_jk(k) = yyy(i_S) - yyy(kk)
        dz_jk(k) = zzz(i_S) - zzz(kk)
      enddo
      call periodic_images(dx_jk(1:i1),dy_jk(1:i1),dz_jk(1:i1))
      dr_sq_jk(1:i1) = dx_jk(1:i1)**2+dy_jk(1:i1)**2+dz_jk(1:i1)**2
      i_r_ijB = 1.0d0/r**beta_ff
      ff1_xx_i = 0.0d0 ; ff1_yy_i = 0.0d0 ; ff1_zz_i = 0.0d0
      ff2_xx_i = 0.0d0 ; ff2_yy_i = 0.0d0 ; ff2_zz_i = 0.0d0
      ff3_xx_i = 0.0d0 ; ff3_yy_i = 0.0d0 ; ff3_zz_i = 0.0d0
      ff2_xx_j = 0.0d0 ; ff2_yy_j = 0.0d0 ; ff2_zz_j = 0.0d0
!s_fk_x=0.0d0;s_fk_y=0.0d0;s_fk_z=0.0d0
      do k = 1,   Next_surf_atom
        kk = Next_Surf(i_S,k)
        x_jk = dx_jk(k) ; y_jk = dy_jk(k) ; z_jk = dz_jk(k)
        r2_jk = dr_sq_jk(k)
        r_jk = dsqrt(r2_jk)
        i_r_jk = 1.0d0/r_jk ; i_r_jk_2 = i_r_jk * i_r_jk
        ps_jk = x*x_jk+y*y_jk+z*z_jk
        i_r_r = 1.0d0/(r_jk*r)
        cos_ijk = -ps_jk * i_r_r
        arg_fi3 = FF_ff*0.5d0*(1.0d0+cos_ijk) ! cos(teha/2)**2
        fi3 = dexp(arg_fi3)
        En = CC * fi3 
        local_energy_repulsion = local_energy_repulsion + En
        CC2 = En*i_C2_r*BB_ff
        ff1_xx_i = ff1_xx_i + CC2 * x  ! for Oxigen
        ff1_yy_i = ff1_yy_i + CC2 * y  
        ff1_zz_i = ff1_zz_i + CC2 * z
        A =  ps_jk*i_r2 
        A1 = ps_jk*i_r_jk_2
        B = En*(FF_ff*0.5d0)*i_r_r
        t2_x = B * ( x_jk - A*x )
        t2_y = B * ( y_jk - A*y )
        t2_z = B * ( z_jk - A*z )
        ff2_xx_i = ff2_xx_i + t2_x
        ff2_yy_i = ff2_yy_i + t2_y
        ff2_zz_i = ff2_zz_i + t2_z
        ff2_xx_k =   B*(-x + A1*x_jk)
        ff2_yy_k =   B*(-y + A1*y_jk)
        ff2_zz_k =   B*(-z + A1*z_jk)
        fxx(kk)= fxx(kk) + ff2_xx_k
        fyy(kk)= fyy(kk) + ff2_yy_k
        fzz(kk)= fzz(kk) + ff2_zz_k
!s_fk_x=s_fk_x+ff2_xx_k;
!s_fk_y=s_fk_y+ff2_yy_k;
!s_fk_z=s_fk_z+ff2_zz_k;
!write(666,*) 'kk:',kk,fxx(kk)
        ff2_xx_j = ff2_xx_j - t2_x - ff2_xx_k
        ff2_yy_j = ff2_yy_j - t2_y - ff2_yy_k
        ff2_zz_j = ff2_zz_j - t2_z - ff2_zz_k
        ff3_xx_i = ff3_xx_i + (beta_i_r2*En) * x
        ff3_yy_i = ff3_yy_i + (beta_i_r2*En) * y
        ff3_zz_i = ff3_zz_i + (beta_i_r2*En) * z
      enddo
    fxx(i_O) = fxx(i_O) + ff1_xx_i  + ff3_xx_i + ff2_xx_i
    fyy(i_O) = fyy(i_O) + ff1_yy_i  + ff3_yy_i + ff2_yy_i
    fzz(i_O) = fzz(i_O) + ff1_zz_i  + ff3_zz_i + ff2_zz_i
    fxx(i_S) = fxx(i_S) - ff1_xx_i  - ff3_xx_i + ff2_xx_j
    fyy(i_S) = fyy(i_S) - ff1_yy_i  - ff3_yy_i + ff2_yy_j
    fzz(i_S) = fzz(i_S) - ff1_zz_i  - ff3_zz_i + ff2_zz_j
!write(666,*)'i_0:',i_O,fxx(i_O)
!write(666,*)'i_S:',i_S,fxx(i_S)
! Surface is assumed rigid and I do not assign any force to it.
    endif
  enddo
!write(666,*) '&&&&&&&&&&&&&&&&&&&&&&&&&&&'
deallocate(dx,dy,dz,dr_sq)
deallocate(dx_jk,dy_jk,dz_jk,dr_sq_jk)




local_energy = local_energy_repulsion + local_energy_attract
!print*, 'atraction repulsion = ',local_energy_attract*1.0d-5,local_energy_repulsion*1.0d-5
en_water_surface_extra = local_energy

!print*, 'max force = ',maxval(dabs(fxx)),maxval(dabs(fyy)),maxval(dabs(fzz))
!print*, 'f=0?',sum(fxx),sum(fyy),sum(fzz)

end subroutine add_force_field

subroutine add_force_field_ENERGY
use mol_utils, only : get_all_mol_dipoles
use boundaries, only : periodic_images
use all_atoms_data, only : xxx,yyy,zzz,fxx,fyy,fzz, all_charges,all_dipoles_xx,all_dipoles_yy,all_dipoles_zz,Natoms,&
                           atom_in_which_molecule
use all_mols_data, only : Nmols, mol_dipole, start_group,end_group
use energies_data
implicit none
integer i,j,k,i_O,i_S,i_mol,i_pair,i1,kk,Next_surf_atom
real(8), allocatable :: dx(:),dy(:),dz(:),dr_sq(:),dx_jk(:),dy_jk(:),dz_jk(:),dr_sq_jk(:)
real(8) x,y,z, qi,qj,qk,dj_xx,dj_yy,dj_zz,r,r2,i_r,i_r2,i_r3,f2,f3,f4,f23,f24,wd,r_wr,i_r_wd,wd2,i_wd2
real(8) ff1_xx,ff1_yy,ff1_zz,ff2_xx,ff2_yy,ff2_zz,ff_xx,ff_yy,ff_zz
real(8) coef5,cos_fi, arg_f3, En, local_energy, i_r_ijB, CC1,CC2, i_C2,i_C2_f2
real(8) f23_cos,fct_cos_fi, r_wd
real(8) x_jk,y_jk,z_jk
real(8) r_jk,r2_jk, i_r_r, cos_ijk,ps_jk,arj_ijk, arg_jk, ps
real(8) fi3,arg_fi3, A, A1, B, beta_i_r2, i_r_jk_2, i_r_jk,pr_i_r2,i_C2_r,pr, i_rij_at_beta
real(8) CC,t2_x,t2_y,t2_z, s_fk_x,s_fk_y,s_fk_z
real(8) local_energy_attract, local_energy_repulsion


allocate(dx(N_pairs),dy(N_pairs),dz(N_pairs),dr_sq(N_pairs))
allocate(dx_jk(MX_Next_SURF),dy_jk(MX_Next_SURF),dz_jk(MX_Next_SURF),dr_sq_jk(MX_Next_SURF))
local_energy_attract = 0.0d0
local_energy_repulsion = 0.0d0
  call get_all_mol_dipoles
  do i_pair = 1, N_pairs
    i_O = which_atom(i_pair)%Ow
    i_S = which_atom(i_pair)%Surf
    dx(i_pair) = xxx(i_O) - xxx(i_S)
    dy(i_pair) = yyy(i_O) - yyy(i_S)
    dz(i_pair) = zzz(i_O) - zzz(I_S)
  enddo
  call periodic_images(dx(1:N_pairs),dy(1:N_pairs),dz(1:N_pairs))
  dr_sq(1:N_pairs) = dx(1:N_pairs)**2+dy(1:N_pairs)**2+dz(1:N_pairs)**2
  do i_pair = 1, N_pairs
    i_O = which_atom(i_pair)%Ow
    i_S = which_atom(i_pair)%Surf
    r2 = dr_sq(i_pair)
    if (r2 <= scut2_ff) then
      r = dsqrt(r2) ; i_r = 1.0d0/r; i_r2 = i_r*i_r ; i_r3 = i_r2*i_r
      x = dx(i_pair) ; y = dy(i_pair) ; z = dz(i_pair)
      f2 = dexp(BB_ff/(r-scut_ff))
      i_mol = atom_in_which_molecule(i_O)
      dj_xx =  mol_dipole(i_mol,1); dj_yy =  mol_dipole(i_mol,2); dj_zz =  mol_dipole(i_mol,3)
      qj = all_charges(i_O)
      pr = dj_xx*x + dj_yy*y + dj_zz*z
      wd2 = dj_xx*dj_xx + dj_yy*dj_yy + dj_zz*dj_zz
      wd = dsqrt(wd2)  ; i_wd2 = 1.0d0/wd2
      r_wd= r*wd
      i_r_wd = 1.0d0/r_wd
      cos_fi = pr*i_r_wd
      arg_f3 = (cos_fi-1.0d0)*0.25d0
      f3 = dexp(-8.0d0*(arg_f3*arg_f3)*(arg_f3*arg_f3))
      f23 = f2*f3
      En = - DD_ff*f23 * i_r3
      local_energy_attract = local_energy_attract + En


! Now add the 3rd body term
      i1 = 0
      i_rij_at_beta = 1.0d0/(r**beta_ff)
      CC = EE_ff*i_rij_at_beta*f2
      beta_i_r2 = beta_ff * i_r2
      Next_surf_atom =  Size_Next_Surf(i_S)
      do k = 1, Next_surf_atom
        kk = Next_Surf(i_S,k)
        i1  = i1 + 1
        dx_jk(k) = xxx(i_S) - xxx(kk)
        dy_jk(k) = yyy(i_S) - yyy(kk)
        dz_jk(k) = zzz(i_S) - zzz(kk)
      enddo
      call periodic_images(dx_jk(1:i1),dy_jk(1:i1),dz_jk(1:i1))
      dr_sq_jk(1:i1) = dx_jk(1:i1)**2+dy_jk(1:i1)**2+dz_jk(1:i1)**2
      i_r_ijB = 1.0d0/r**beta_ff
      do k = 1,   Next_surf_atom
        kk = Next_Surf(i_S,k)
        x_jk = dx_jk(k) ; y_jk = dy_jk(k) ; z_jk = dz_jk(k)
        r2_jk = dr_sq_jk(k)
        r_jk = dsqrt(r2_jk)
        i_r_jk = 1.0d0/r_jk ; i_r_jk_2 = i_r_jk * i_r_jk
        ps_jk = x*x_jk+y*y_jk+z*z_jk
        i_r_r = 1.0d0/(r_jk*r)
        cos_ijk = -ps_jk * i_r_r
        arg_fi3 = FF_ff*0.5d0*(1.0d0+cos_ijk) ! cos(teha/2)**2
        fi3 = dexp(arg_fi3)
        En = CC * fi3
        local_energy_repulsion = local_energy_repulsion + En
      enddo
    endif
  enddo
!write(666,*) '&&&&&&&&&&&&&&&&&&&&&&&&&&&'
deallocate(dx,dy,dz,dr_sq)
deallocate(dx_jk,dy_jk,dz_jk,dr_sq_jk)




local_energy = local_energy_repulsion + local_energy_attract
!print*, 'atraction repulsion = ',local_energy_attract*1.0d-5,local_energy_repulsion*1.0d-5
en_water_surface_extra = local_energy

!print*, 'max force = ',maxval(dabs(fxx)),maxval(dabs(fyy)),maxval(dabs(fzz))
!print*, 'f=0?',sum(fxx),sum(fyy),sum(fzz)

end subroutine add_force_field_ENERGY

 end module water_Pt_ff_module
