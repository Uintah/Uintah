module non_bonded_lists_builder
implicit none

integer, private :: ii_xx, ii_yy, ii_zz
real(8), private :: local_cut, i_dr, local_cut_sq, local_cut_short, local_cut_short_sq

! methods
public :: driver_link_nonbonded_pairs
private :: simple_link_long_OZ_MOLCUT
private :: simple_link_MOLCUT
private :: simple_link_long_OZ
private :: simple_link
private :: link_pairs_by_linked_cells

contains

subroutine driver_link_nonbonded_pairs(l_update)
use sim_cel_data
use cut_off_data
use integrate_data, only : l_do_QN_CTRL
use all_atoms_data, only : Natoms
use all_mols_data, only : Nmols
use paralel_env_data
use sys_preparation_data, only : sys_prep

logical, intent(IN) :: l_update
logical l_simple_lincs,long_z
real(8) dx,dy,dz

if (.not.l_update) RETURN

 local_cut = cut_off + displacement
 local_cut_sq = local_cut*local_cut
 local_cut_short = cut_off_short + displacement
 local_cut_short_sq = local_cut_short*local_cut_short



if (l_do_QN_CTRL) then
   long_z = cel_cos_c_axb/max(cel_cos_a_bxc,cel_cos_b_axc) > 1.5d0

if (sys_prep%any_prep) then
  call  simple_link_MOLCUT
  RETURN
endif !sys_prep%any_prep


   if (long_z) then
     call simple_link_long_OZ_MOLCUT
   else
     call simple_link_MOLCUT
   endif
RETURN
endif
 
     i_dr = 1.0d0/local_cut
     dx = cel_cos_a_bxc*i_dr  ;  dy = cel_cos_b_axc*i_dr   ; dz = cel_cos_c_axb*i_dr
     ii_xx=int(dx) 
     ii_yy=int(dy)
     ii_zz=int(dz)
     l_simple_lincs = ii_xx < 3 .or. ii_yy < 3 .or. ii_zz < 3 

if (sys_prep%any_prep) then
  call  simple_link
  RETURN
endif !sys_prep%any_prep


     if (Natoms < 4000.or.l_simple_lincs) then
       long_z = dz/max(dx,dy) > 1.5d0
       if (long_z) then
           call simple_link_LONG_OZ
           RETURN
       else
           call simple_link
           RETURN
       endif ! long_z and simple_link
     else  ! do linked cells
           call link_pairs_by_linked_cells
           RETURN
     endif 

end  subroutine driver_link_nonbonded_pairs

!----------------------------------------------

subroutine simple_link_long_OZ_MOLCUT
use sim_cel_data
use boundaries
use non_bonded_lists_data
use paralel_env_data
use ALL_atoms_data, only : Natoms,xxx,yyy,zzz,l_WALL, xx, yy, zz, atom_in_which_molecule
use ALL_mols_data, only : Nmols, start_group, end_group, mol_xyz
use connectivity_ALL_data, only : list_excluded_HALF,size_list_excluded_HALF
use cut_off_data
use mol_type_data, only : N_type_atoms_per_mol_type
implicit none
 integer i,j,k,ii,jj,imol,jmol,ijk,i1, isgi, iegi, i2, k1,k2
 real(8) , allocatable :: dx(:),dy(:),dz(:),dr_sq(:), ddz(:)
 integer, allocatable :: in_list(:)
 integer  MX_size

 MX_size = Natoms*maxval(N_type_atoms_per_mol_type)

allocate(in_list(MX_size))
allocate(dx(MX_size),dy(MX_size),dz(MX_size),dr_sq(MX_size),ddz(MX_size))

 size_list_nonbonded=0
 if (set_2_nonbonded_lists_CTRL) size_list_nonbonded_short=0

  do i = 1, Nmols-1
  i1 = 0
  isgi = start_group(i)
  iegi = end_group(i)
  do j = i+1, Nmols
   i1  =  i1 + 1
   if (i1 > MX_size) then
      print*, 'ERROR in simple_link_long_OZ_MOLCUT i1 > MX_size', i1,MX_size, ' increase MX_size (the size of local arraya)'
      STOP
   endif
   dz(i1) = mol_xyz(i,3) - mol_xyz(j,3)
  enddo   ! jj
  call periodic_images_zz(dz(1:i1))
  i2 = 0
  i1=0
  do j = i+1, Nmols
   i1 = i1 + 1
   if (dz(i1) < local_cut) then
     i2  =  i2 + 1
     dx(i2) = mol_xyz(i,1) - mol_xyz(j,1)
     dy(i2) = mol_xyz(i,2) - mol_xyz(j,2)
     ddz(i2) = dz(i1)
     in_list(i2) = j
   endif
  enddo   ! jj
  call periodic_images_xx_yy(dx(1:i2),dy(1:i2))
  dr_sq(1:i2) = dx(1:i2)*dx(1:i2)+dy(1:i2)*dy(1:i2)+ddz(1:i2)*ddz(1:i2)
  k2 = 0
  do k = 1, i2
   if (dr_sq(k)<local_cut_sq) then
     j = in_list(k)
     do ii =  isgi,iegi 
     do jj =  start_group(j), end_group(j)
         size_list_nonbonded(ii) = size_list_nonbonded(ii) + 1
         if(size_list_nonbonded(ii) > MX_list_nonbonded)then
               write(6,*) rank,'ERROR in link_pairs; size_list_nonbonded_short(i1) > MX_list_nonbonded_short',&
                           size_list_nonbonded(ii),MX_list_nonbonded
               write(6,*) 'Increase MX_list_nonbonded_SHORT or define in input file DENS_VAR > 1'
               STOP
         endif
         list_nonbonded(ii,size_list_nonbonded(ii)) = jj
                              if (set_2_nonbonded_lists_CTRL) then
                                if (dr_sq(k) < local_cut_short_sq) then
                                     size_list_nonbonded_short(ii) = size_list_nonbonded_short(ii) + 1
                                if(size_list_nonbonded_short(ii) > MX_list_nonbonded_short)then
    write(6,*) rank,'ERROR in link_pairs; size_list_nonbonded_short(i1) > MX_list_nonbonded_short',&
               size_list_nonbonded_short(ii),MX_list_nonbonded_short
    write(6,*) 'Increase MX_list_nonbonded_SHORT or define in input file DENS_VAR > 1'
    STOP
                                 else
                                     list_nonbonded_short(ii,size_list_nonbonded_short(ii))=jj
                                 endif
                                 endif
                            endif ! set_2_nonbonded_lists_CTRL
     enddo ! ii
     enddo ! ii
   endif ! (dr_sq(k)<local_cut_sq)
  enddo
  
 enddo ! i
 
end subroutine simple_link_long_OZ_MOLCUT
!-----------------------------------------------------------------------
subroutine simple_link_MOLCUT
use sim_cel_data
use boundaries
use non_bonded_lists_data
use paralel_env_data
use ALL_atoms_data, only : Natoms,xxx,yyy,zzz,l_WALL, xx, yy, zz, atom_in_which_molecule
use ALL_mols_data, only : Nmols, start_group, end_group, mol_xyz
use connectivity_ALL_data, only : list_excluded_HALF,size_list_excluded_HALF
use cut_off_data
use mol_type_data, only : N_type_atoms_per_mol_type
implicit none
 integer i,j,k,ii,jj,imol,jmol,ijk,i1, isgi, iegi, i2, k1,k2 
 real(8) , allocatable :: dx(:),dy(:),dz(:),dr_sq(:), ddz(:)
 integer, allocatable :: in_list(:)
 integer MX_size

MX_size = Natoms*maxval(N_type_atoms_per_mol_type)

allocate(in_list(Natoms))
allocate(dx(Natoms),dy(Natoms),dz(Natoms),dr_sq(Natoms),ddz(Natoms))

 size_list_nonbonded=0
 if (set_2_nonbonded_lists_CTRL) size_list_nonbonded_short=0

 do i = 1, Nmols-1
  i1 = 0
  isgi = start_group(i)
  iegi = end_group(i)
  do j = i+1, Nmols
  i1  =  i1 + 1
     if (i1 > MX_size) then
      print*, 'ERROR in simple_link_MOLCUT i1 > MX_size', i1,MX_size, ' increase MX_size (the size of local arraya)'
      STOP
   endif 
  dx(i1) = mol_xyz(i,1) - mol_xyz(j,1)
  dy(i1) = mol_xyz(i,2) - mol_xyz(j,2)
  dz(i1) = mol_xyz(i,3) - mol_xyz(j,3)
  in_list(i1) = j
  enddo   ! jj
  call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
  dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
  i2  =  0
  do k = 1 , i1
    if (dr_sq(k) < local_cut_sq) then
      j = in_list(k)
      do ii = isgi, iegi
      do jj = start_group(j), end_group(j)
         size_list_nonbonded(ii) = size_list_nonbonded(ii) + 1
         if(size_list_nonbonded(ii) > MX_list_nonbonded)then
               write(6,*) rank,'ERROR in link_pairs; size_list_nonbonded_short(i1) > MX_list_nonbonded_short',&
                           size_list_nonbonded(ii),MX_list_nonbonded
               write(6,*) 'Increase MX_list_nonbonded_SHORT or define in input file DENS_VAR > 1'
               STOP
         endif
         list_nonbonded(ii,size_list_nonbonded(ii)) = jj
                              if (set_2_nonbonded_lists_CTRL) then
                                if (dr_sq(k) < local_cut_short_sq) then
                                     size_list_nonbonded_short(ii) = size_list_nonbonded_short(ii) + 1
                                if(size_list_nonbonded_short(ii) > MX_list_nonbonded_short)then
    write(6,*) rank,'ERROR in link_pairs; size_list_nonbonded_short(i1) > MX_list_nonbonded_short',&
               size_list_nonbonded_short(ii),MX_list_nonbonded_short
    write(6,*) 'Increase MX_list_nonbonded_SHORT or define in input file DENS_VAR > 1'
    STOP
                                 else
                                     list_nonbonded_short(ii,size_list_nonbonded_short(ii))=jj
                                 endif
                                 endif
                            endif ! set_2_nonbonded_lists_CTRL
         enddo ! ii
      enddo ! jj
     endif !
    enddo ! k = 1, i1 
 enddo ! i

deallocate(in_list)
deallocate(dx,dy,dz,dr_sq)

end subroutine simple_link_MOLCUT

subroutine simple_link_LONG_OZ
use sim_cel_data
use boundaries
use non_bonded_lists_data
use paralel_env_data
use ALL_atoms_data, only : Natoms,xxx,yyy,zzz,l_WALL, xx, yy, zz, atom_in_which_molecule
use connectivity_ALL_data, only : list_excluded_HALF,size_list_excluded_HALF
use cut_off_data

implicit none
 integer i,j,k,ii,jj,imol,jmol,ijk,i1,i2,k2, k1
 real(8) , allocatable :: dx(:),dy(:),dz(:),ddz(:),dr_sq(:)
 integer, allocatable :: in_list(:)
 logical l_proceed

 allocate(dx(Natoms),dy(Natoms),dz(Natoms),ddz(Natoms),dr_sq(Natoms))
 allocate(in_list(Natoms))
 size_list_nonbonded=0
 if (set_2_nonbonded_lists_CTRL) size_list_nonbonded_short=0

 do i = 1, Natoms-1
   k = 0
   imol = atom_in_which_molecule(i)
   do j = i+1, Natoms
     k = k + 1
     dz(k) = zzz(i) - zzz(j)
   enddo ! j
   if (k>0) call periodic_images_zz(dz(1:k))
   k = 0
   k1 = 0
   do j = i+1, Natoms
     k = k + 1
     if (dz(k) < local_cut) then
        l_proceed = .true.
        if (imol == atom_in_which_molecule(j) ) then
        do ijk=1,size_list_excluded_HALF(i)
              if(list_excluded_HALF(i,ijk).eq.j) then
                 l_proceed = .false.
                           GOTO 3
              endif
        enddo
        endif
3   CONTINUE
       if (l_proceed) then
          k1 = k1 + 1
          dx(k1) = xxx(i) - xxx(j)
          dy(k1) = yyy(i) - yyy(j)
          ddz(k1) = dz(k)
          in_list(k1) = j
       endif
     endif  ! dz(k) < local_cut
   enddo ! j
   if (k1>0) call periodic_images_xx_yy(dx(1:k1),dy(1:k1))
   dr_sq(1:k1) = dx(1:k1)*dx(1:k1)+dy(1:k1)*dy(1:k1)+ddz(1:k1)*ddz(1:k1)
   k2 = 0
   do k = 1, k1
      if (dr_sq(k) < local_cut_sq) then
         k2 = k2 + 1
         if(k2 > MX_list_nonbonded)then
               write(6,*) 'ERROR in link_pairs; size_list_nonbonded(i1) > MX_list_nonbonded',&
                          k2,MX_list_nonbonded
               write(6,*) 'Increase MX_list_nonbonded or define in input file DENS_VAR > 1'
               STOP
         endif
         j = in_list(k)
         size_list_nonbonded(i) =  size_list_nonbonded(i) + 1
         list_nonbonded(i, k2) = j
                               if (set_2_nonbonded_lists_CTRL) then
                                if (dr_sq(k) < local_cut_short_sq) then
                                     size_list_nonbonded_short(i) = size_list_nonbonded_short(i) + 1
                                if(size_list_nonbonded_short(i) > MX_list_nonbonded_short)then
    write(6,*) rank,'ERROR in link_pairs; size_list_nonbonded_short(i1) > MX_list_nonbonded_short',&
               size_list_nonbonded_short(i),MX_list_nonbonded_short
    write(6,*) 'Increase MX_list_nonbonded_SHORT or define in input file DENS_VAR > 1'
    STOP
                                 else
                                     list_nonbonded_short(i,size_list_nonbonded_short(i))=j
                                 endif
                                 endif
                            endif ! set_2_nonbonded_lists_CTRL

      endif
   enddo
   
 enddo ! i
 
 deallocate(dx,dy,dz,ddz,dr_sq)
 deallocate(in_list)

end subroutine simple_link_LONG_OZ


subroutine simple_link

use sim_cel_data
use boundaries
use non_bonded_lists_data
use paralel_env_data
use ALL_atoms_data, only : Natoms,xxx,yyy,zzz,l_WALL, xx, yy, zz, atom_in_which_molecule,&
                           is_sfield_constrained, is_dipole_polarizable
use connectivity_ALL_data, only : list_excluded_HALF,size_list_excluded_HALF
use cut_off_data
use cg_buffer, only : use_cg_preconditioner_CTRL
use preconditioner_data

implicit none
 integer i,j,k,ii,jj,imol,jmol,ijk,i1
 real(8) , allocatable :: dx(:),dy(:),dz(:),dr_sq(:)
 integer, allocatable :: in_list(:)
 logical l_proceed


 allocate(dx(Natoms),dy(Natoms),dz(Natoms),dr_sq(Natoms))
 allocate(in_list(Natoms))

 size_list_nonbonded(Natoms) = 0
 if (set_2_nonbonded_lists_CTRL) size_list_nonbonded_short=0
 if (use_cg_preconditioner_CTRL) size_preconditioner=0
 do i = 1, Natoms  - 1
   k = 0
   imol = atom_in_which_molecule(i)
   do j = i+1, Natoms
   k = k + 1
     dx(k) = xxx(i) - xxx(j)
     dy(k) = yyy(i) - yyy(j)
     dz(k) = zzz(i) - zzz(j)
     in_list(k) = j
   enddo ! j
   i1 = k
   call periodic_images(dx(1:i1),dy(1:i1),dz(1:i1))
   dr_sq(1:i1) = dx(1:i1)*dx(1:i1) + dy(1:i1)*dy(1:i1) + dz(1:i1)*dz(1:i1)
   k = 0
   do jj = 1 , i1
     if (dr_sq(jj) < local_cut_sq) then
       j = in_list(jj)
       jmol = atom_in_which_molecule(j)
       l_proceed = .true.
       if (imol == jmol) then
       do ijk=1,size_list_excluded_HALF(i)
              if(list_excluded_HALF(i,ijk).eq.j) then
                 l_proceed = .false.
                           GOTO 3
              endif
        enddo
3 CONTINUE
       endif
       if (l_proceed) then
                               if (set_2_nonbonded_lists_CTRL) then
                                if (dr_sq(jj) < local_cut_short_sq) then
                                     size_list_nonbonded_short(i) = size_list_nonbonded_short(i) + 1
                                if(size_list_nonbonded_short(i) > MX_list_nonbonded_short)then
    write(6,*) rank,'ERROR in link_pairs; size_list_nonbonded_short(i1) > MX_list_nonbonded_short',&
               size_list_nonbonded_short(i),MX_list_nonbonded_short
    write(6,*) 'Increase MX_list_nonbonded_SHORT or define in input file DENS_VAR > 1'
    STOP
                                 else
                                     list_nonbonded_short(i,size_list_nonbonded_short(i))=j
                                 endif
                                 endif
                            endif ! set_2_nonbonded_lists_CTRL
                            if (use_cg_preconditioner_CTRL) then
                            if (is_sfield_constrained(i).or.is_dipole_polarizable(i)) then
                              if (dr_sq(jj) < preconditioner_cut_off_sq) then
                              if (is_sfield_constrained(j).or.is_dipole_polarizable(j)) then
                                size_preconditioner(i) = size_preconditioner(i) + 1
                                if (size_preconditioner(i) > MX_preconditioner_size) then
print*, 'ERROR in link_pairs%simple_links (size_preconditioner(i) > MX_preconditioner_size',&
size_preconditioner(i),MX_preconditioner_size,&
'Increase the MX_preconditioner_size in header_list, recomplie and restart'
STOP
                                else
                                  preconditioner_rr(i,size_preconditioner(i)) = dsqrt(dr_sq(jj))
                                  preconditioner_xx(i,size_preconditioner(i)) = dx(jj)
                                  preconditioner_yy(i,size_preconditioner(i)) = dy(jj)
                                  preconditioner_zz(i,size_preconditioner(i)) = dz(jj)
                                endif ! size_preconditioner(i) > MX_preconditioner_size
                              endif 
                              endif ! (dr_sq(jj) < preconditioner_cut_off_sq)
                            endif
                            endif ! use_preconditioner_CTRL

         k = k + 1 
       if(k > MX_list_nonbonded)then
    write(6,*) 'ERROR in link_pairs; size_list_nonbonded(i1) > MX_list_nonbonded',&
               k,MX_list_nonbonded
    write(6,*) 'Increase MX_list_nonbonded or define in input file DENS_VAR > 1'
    STOP
       endif
         list_nonbonded(i,k) = j
       endif
     endif
   enddo 
   size_list_nonbonded(i) = k
 enddo ! i = 1, Natoms -1
!do i = 1, size_list_nonbonded(1)
!print*,i,'list=', list_nonbonded(1,i)
!read(*,*)
!enddo

 deallocate(dx,dy,dz,dr_sq)
 deallocate(in_list)

end subroutine simple_link



!subroutine distribute_lists(l_update)
!logical, intent(IN) :: l_update
!
!   if(.not.l_update) RETURN
!
!   do i = 1, Natoms
!    i1 = size_list_nonbonded(i)
!    l_i = is_charge_distributed(i)
!    do k = 1, i1
!       j = list_nonbonded(i,k)
!       l_j = is_charge_distributed(j)
!       k1 = 0 ; k2 = 0 ; k3 = 0 ; k4 = 0
!       if (l_i.and.l_j) then
!        k1 = k1 + 1
!        list_nonb_G_G(i,k1) = j
!       else if ((.not.l_i).and.l_j) then
!        k2 = k2 + 1
!        list_nonb_P_G(i,k2) = j
!       else if (l_i.and.(.not.l_j)) then
!        k3 = k3 + 1
!        list_nonb_G_P(i,k3) = j
!       else ! not li and not lj (point - point
!        k4 = k4 + 1
!        list_nonb_P_P(i,k4) = j
!       endif
!       size_list_nonb_G_G(i) = k1
!       size_list_nonb_P_G(i) = k2
!       size_list_nonb_G_P(i) = k3
!       size_list_nonb_P_P(i) = k4
!    enddo
!   enddo
!
!end subroutine distribute_lists


subroutine link_pairs_by_linked_cells
use sim_cel_data
use boundaries
use non_bonded_lists_data
use paralel_env_data
use ALL_atoms_data, only : Natoms,xxx,yyy,zzz,l_WALL, xx, yy, zz, atom_in_which_molecule
use connectivity_ALL_data, only : list_excluded_HALF,size_list_excluded_HALF,&
                                  list_excluded,size_list_excluded
use cut_off_data
!use sizes_data, only :  Natoms

implicit none
logical l_update,linc,l_do,ldo,l_WALL_i,l_WALL_ij
integer, save :: nnx(508),nny(508),nnz(508)
logical, save :: l_very_first_run=.true.
integer i_ratio,n_subcells, icell,jcell,jcell_0
integer i,j,i1,k
real(8) cel1,cel2,cel3,cel4,cel5,cel6,cel7,cel8,cel9
real(8) ivc1,ivc2,ivc3,ivc4,ivc5,ivc6,ivc7,ivc8,ivc9
real(8) sx,sy,sz,x,y,z,rsq
integer ix,iy,iz,ilnkat_i,ilnkat_j,ijk,jx,jy,jz
real(8) side_xx,side_yy,side_zz

real(8), allocatable :: tx(:),ty(:),tz(:)  ! reduced coordinates

allocate(tx(Natoms),ty(Natoms),tz(Natoms))

 if (l_very_first_run) then 
   MX_cells = (ii_xx+1)*(ii_yy+1)*(ii_zz+1)
   allocate (link_cell_to_atom(MX_cells)) 
   call get_nnxyz_arrays  
   l_very_first_run=.false.
 endif
 
 call initializations

ix = 1; iy = 1; iz = 1;
! start the mess : cycle over cells

 do icell = 1, Ncells
!print*, 'icell=',icell
   ilnkat_i = link_cell_to_atom(icell)
   if (ilnkat_i > 0) then
     i1 = 0
     do jcell = 1, n_subcells
!print*, 'jcell=',jcell
      i = ilnkat_i
      side_xx=0.d0
      side_yy=0.d0
      side_zz=0.d0
      jx=ix+nnx(jcell)
      jy=iy+nny(jcell)
      jz=iz+nnz(jcell)
!     minimum image convention              
      if(jx.gt.ii_xx)then                
          jx=jx-ii_xx
          side_xx=1.d0                
      elseif(jx.lt.1)then                
          jx=jx+ii_xx
          side_xx=-1.d0                
      endif              
      if(jy.gt.ii_yy)then                
          jy=jy-ii_yy
          side_yy=1.d0                
      elseif(jy.lt.1)then
          jy=jy+ii_yy
          side_yy=-1.d0                
      endif              
      if(jz.gt.ii_zz)then                
          jz=jz-ii_zz
          side_zz=1.d0                
      elseif(jz.lt.1)then               
          jz=jz+ii_zz
          side_zz=-1.d0                
      endif

      jcell_0=jx+ii_xx*((jy-1)+ii_yy*(jz-1))
      j=link_cell_to_atom(jcell_0)
      if (j > 0) then
        do while (i /= 0)
          if(mod(i-1,nprocs).eq.rank)then  ! if this the node we are interested in?
            i1 = ((i-1)/nprocs)+1
            l_WALL_i = l_WALL(i)
            if(icell.eq.jcell_0) j=link_atom_to_cell(i)
            if (j > 0) then

            do while ( j/=0 )
              if (l_WALL_i) then
               l_WALL_ij = l_WALL_i.and.l_WALL(j)
               l_do =.not.l_WALL_ij
              else
                l_do = .true.
              endif

              if (l_do) then
                  sx=tx(j)-tx(i)+side_xx
                  sy=ty(j)-ty(i)+side_yy
                  sz=tz(j)-tz(i)+side_zz                         
                  x=cel1*sx+cel4*sy+cel7*sz
                  y=cel2*sx+cel5*sy+cel8*sz
                  z=cel3*sx+cel6*sy+cel9*sz
                  rsq=x**2+y**2+z**2        
                  if (rsq < local_cut_sq)then
                      linc=.true.
                       if (atom_in_which_molecule(i1)==atom_in_which_molecule(j)) then
                        do ijk=1,size_list_excluded(i1) !size_list_excluded_HALF(i1)                              
                          if(list_excluded(i1,ijk)==j)then!(list_excluded_HALF(i1,ijk).eq.j) then 
!! I put list excluded becasue I either need tho whole list or a different kind of structure of
! list excluded_HALF .
! size_list_excluded and list_excluded must be updated accorinf to actual content of theur *HALF counterparts

                                linc=.false.    
GOTO 3
                          endif
                        enddo      
3 CONTINUE
                       endif ! (atom_in_which_molecule(i1)==atom_in_which_molecule(j)

                        if(linc)then                              
                           if (set_2_nonbonded_lists_CTRL) then 
                                if (rsq < local_cut_short_sq) then 
                                     size_list_nonbonded_short(i1) = size_list_nonbonded_short(i1) + 1
                                if(size_list_nonbonded_short(i1) > MX_list_nonbonded_short)then
    write(6,*) rank,'ERROR in link_pairs; size_list_nonbonded_short(i1) > MX_list_nonbonded_short',&
               size_list_nonbonded_short(i1),MX_list_nonbonded_short
    write(6,*) 'Increase MX_list_nonbonded_SHORT or define in input file DENS_VAR > 1'
    STOP
                                 else
                                     list_nonbonded_short(i1,size_list_nonbonded_short(i1))=j
                                 endif
                                 endif
                            endif ! set_2_nonbonded_lists_CTRL
                              size_list_nonbonded(i1)=size_list_nonbonded(i1)+1                              
                              if(size_list_nonbonded(i1) > MX_list_nonbonded)then                                
    write(6,*) rank,'ERROR in link_pairs; size_list_nonbonded(i1) > MX_list_nonbonded',&
               size_list_nonbonded(i1),MX_list_nonbonded
    write(6,*) 'Increase MX_list_nonbonded or define in input file DENS_VAR > 1'
    STOP                                                          
                              else                                
                               list_nonbonded(i1,size_list_nonbonded(i1))=j                                

                              endif                              
                        endif ! link
                  endif ! rsq < local_cut_sq
              endif ! l_do
             j = link_atom_to_cell(j)
            enddo ! while j =/0

          endif !
          endif
          j=link_cell_to_atom(jcell_0)
          i=link_atom_to_cell(i)                  
        enddo ! while (i /= 0)
      endif ! ilnkat_j > 0)
     enddo ! jcell = 1, N_subcell (second loop over cells)
     endif ! (ilnkat_i > 0) then
       ix=ix+1
       if(ix.gt.ii_xx)then
       ix=1
       iy=iy+1            
       if(iy.gt.ii_yy)then              
          iy=1
          iz=iz+1              
       endif           
       endif
 enddo ! main cycle icell = 1, Ncells

! end the mess

deallocate(tx,ty,tz)


contains 
 subroutine initializations
 use boundaries, only : adjust_box
  real(8) xmax,ymax,zmax
  integer iii,ix,iy,iz,icell
  logical l_OK
  real(8) x,y,z

   i_ratio = 1 
   select case (i_ratio)
    case(1) 
        n_subcells = 14
    case(2)
        n_subcells = 63
    case(3)
        n_subcells = 156
    case(4)
        n_subcells = 307
    case(5)
        n_subcells = 508
   end select

!    i_dr = 1.0d0/(local_cut)
if (i_ratio > 1 ) then   
     ii_xx=int(cel_cos_a_bxc*dble(i_ratio)*i_dr)
     ii_yy=int(cel_cos_b_axc*dble(i_ratio)*i_dr)
     ii_zz=int(cel_cos_c_axb*dble(i_ratio)*i_dr)
else   ! done already
!     ii_xx=int(cel_cos_a_bxc*i_dr)
!     ii_yy=int(cel_cos_b_axc*i_dr)
!     ii_zz=int(cel_cos_c_axb*i_dr)
endif

     iii = 2*i_ratio+1
     if (ii_xx < iii) then
        print*, 'ERROR in list builder%initialize ; ii_xx < iii; too small OX box size or too big cut off',ii_xx,iii
        STOP
     endif
     if (ii_yy < iii) then
        print*, 'ERROR in list builder%initialize ; ii_yy < iii; too small OY box size or too big cut off',ii_yy,iii
        STOP 
     endif
     if (ii_zz < iii) then
        print*, 'ERROR in list builder%initialize ; ii_zz < iii; too small OZ box size or too big cut off',ii_zz,iii
        STOP
     endif

     Ncells=ii_xx*ii_yy*ii_zz
     if(ncells > MX_cells) then 
         write(6,*) 'ERROR in initializations of link_pairs'
         write(6,*) 'Ncells > MX_cell ; Increase MX_cell ', Ncells, MX_cells
         STOP
     endif
   size_list_nonbonded = 0 ; 
   if (set_2_nonbonded_lists_CTRL) size_list_nonbonded_short=0
   link_cell_to_atom = 0
   link_atom_to_cell = 0
   ! link_atom_to_cell(1:Natoms)
   ivc1 = inverse_cel(1); ivc2 = inverse_cel(2); ivc3 = inverse_cel(3)
   ivc4 = inverse_cel(4); ivc5 = inverse_cel(5); ivc6 = inverse_cel(6)
   ivc7 = inverse_cel(7); ivc8 = inverse_cel(8); ivc9 = inverse_cel(9)
   cel1 = sim_cel(1) ; cel2 = sim_cel(2) ; cel3 = sim_cel(3)
   cel4 = sim_cel(4) ; cel5 = sim_cel(5) ; cel6 = sim_cel(6)
   cel7 = sim_cel(7) ; cel8 = sim_cel(8) ; cel9 = sim_cel(9)
   call adjust_box
   do i=1, Natoms         
          x=xx(i)
          y=yy(i)
          z=zz(i)
          tx(i)=(ivc1*x+ivc4*y+ivc7*z)+0.5d0
          ty(i)=(ivc2*x+ivc5*y+ivc8*z)+0.5d0
          tz(i)=(ivc3*x+ivc6*y+ivc9*z)+0.5d0          
   enddo
   do i = 1, Natoms
          ix=min(int(dble(ii_xx)*tx(i)),ii_xx-1)
          iy=min(int(dble(ii_yy)*ty(i)),ii_yy-1)
          iz=min(int(dble(ii_zz)*tz(i)),ii_zz-1)         
          icell=1+ix+ii_xx*(iy+ii_yy*iz)          
          j=link_cell_to_atom(icell) ! if not linked then it 0
          link_cell_to_atom(icell)=i
          link_atom_to_cell(i)=j
    enddo
   
 end subroutine initializations

 
 subroutine  get_nnxyz_arrays
 
 nnx = (/ 0,1,0,0,-1,1,0,-1,1,0,-1,1,-1,1,2,0,0,-2,2,-1,1,0,-2,2,0,&
     0,-1,1,0,-1,1,-2,2,-2,2,-1,1,-1,1,-1,1,-2,2,0,-2,2,0,-2,2,-2,2,&
     -1,1,-2,2,-2,2,-1,1,-2,2,-2,2,3,0,0,-3,3,-1,1,0,-3,3,0,0,-1,1,0,&
     -1,1,-3,3,-3,3,-1,1,-1,1,-1,1,-3,3,-2,2,0,-3,3,0,0,-2,2,0,-2,2,&
     -3,3,-3,3,-2,2,-1,1,-3,3,-3,3,-1,1,-1,1,-2,2,-2,2,-1,1,-2,2,-3,3,&
     -3,3,-2,2,-2,2,-2,2,-3,3,0,-3,3,0,-3,3,-3,3,-1,1,-3,3,-3,3,-1,1,&
     -3,3,-3,3,-2,2,-3,3,-3,3,-2,2,-3,3,-3,3,4,0,0,-4,4,-1,1,0,-4,4,0,&
     0,-1,1,0,-1,1,-4,4,-4,4,-1,1,-1,1,-1,1,-4,4,-2,2,0,-4,4,0,0,-2,2,&
     0,-2,2,-4,4,-4,4,-2,2,-1,1,-4,4,-4,4,-1,1,-1,1,-2,2,-2,2,-1,1,-2,&
     2,-4,4,-4,4,-2,2,-2,2,-2,2,-4,4,-3,3,0,-4,4,0,0,-3,3,0,-3,3,-4,4,&
     -4,4,-3,3,-1,1,-4,4,-4,4,-1,1,-1,1,-3,3,-3,3,-1,1,-3,3,-4,4,-4,4,&
     -3,3,-2,2,-4,4,-4,4,-2,2,-2,2,-3,3,-3,3,-2,2,-3,3,-4,4,-4,4,-3,3,&
     -3,3,-3,3,-4,4,0,-4,4,0,-4,4,-4,4,-1,1,-4,4,-4,4,-1,1,-4,4,-4,4,&
     -2,2,-4,4,-4,4,-2,2,-4,4,-4,4,-3,3,-4,4,-4,4,-3,3,5,0,0,-5,5,-1,&
     1,0,-5,5,0,0,-1,1,0,-1,1,-5,5,-5,5,-1,1,-1,1,-1,1,-5,5,-2,2,0,-5,&
     5,0,0,-2,2,0,-2,2,-5,5,-5,5,-2,2,-1,1,-5,5,-5,5,-1,1,-1,1,-2,2,&
     -2,2,-1,1,-2,2,-5,5,-5,5,-2,2,-2,2,-2,2,-5,5,-3,3,0,-5,5,0,0,-3,&
     3,0,-3,3,-5,5,-5,5,-3,3,-1,1,-5,5,-5,5,-1,1,-1,1,-3,3,-3,3,-1,1,&
     -3,3,-5,5,-5,5,-3,3,-2,2,-5,5,-5,5,-2,2,-2,2,-3,3,-3,3,-2,2,-3,3,&
     -5,5,-5,5,-3,3,-3,3,-3,3 /)
 nny = (/  0,0,1,0,1,1,-1,0,0,1,-1,-1,1,1,0,2,0,1,1,2,2,-2,0,0,2,&
     -1,0,0,1,-2,-2,-1,-1,1,1,2,2,-1,-1,1,1,2,2,-2,0,0,2,-2,-2,2,2,-2,&
     -2,-1,-1,1,1,2,2,-2,-2,2,2,0,3,0,1,1,3,3,-3,0,0,3,-1,0,0,1,-3,-3,&
     -1,-1,1,1,3,3,-1,-1,1,1,2,2,3,3,-3,0,0,3,-2,0,0,2,-3,-3,-2,-2,2,&
     2,3,3,-3,-3,-1,-1,1,1,3,3,-2,-2,-1,-1,1,1,2,2,-3,-3,-2,-2,2,2,3,&
     3,-2,-2,2,2,3,3,-3,0,0,3,-3,-3,3,3,-3,-3,-1,-1,1,1,3,3,-3,-3,3,3,&
     -3,-3,-2,-2,2,2,3,3,-3,-3,3,3,0,4,0,1,1,4,4,-4,0,0,4,-1,0,0,1,-4,&
     -4,-1,-1,1,1,4,4,-1,-1,1,1,2,2,4,4,-4,0,0,4,-2,0,0,2,-4,-4,-2,-2,&
     2,2,4,4,-4,-4,-1,-1,1,1,4,4,-2,-2,-1,-1,1,1,2,2,-4,-4,-2,-2,2,2,&
     4,4,-2,-2,2,2,3,3,4,4,-4,0,0,4,-3,0,0,3,-4,-4,-3,-3,3,3,4,4,-4,&
     -4,-1,-1,1,1,4,4,-3,-3,-1,-1,1,1,3,3,-4,-4,-3,-3,3,3,4,4,-4,-4,&
     -2,-2,2,2,4,4,-3,-3,-2,-2,2,2,3,3,-4,-4,-3,-3,3,3,4,4,-3,-3,3,3,&
     4,4,-4,0,0,4,-4,-4,4,4,-4,-4,-1,-1,1,1,4,4,-4,-4,4,4,-4,-4,-2,-2,&
     2,2,4,4,-4,-4,4,4,-4,-4,-3,-3,3,3,4,4,0,5,0,1,1,5,5,-5,0,0,5,-1,&
     0,0,1,-5,-5,-1,-1,1,1,5,5,-1,-1,1,1,2,2,5,5,-5,0,0,5,-2,0,0,2,-5,&
     -5,-2,-2,2,2,5,5,-5,-5,-1,-1,1,1,5,5,-2,-2,-1,-1,1,1,2,2,-5,-5,&
     -2,-2,2,2,5,5,-2,-2,2,2,3,3,5,5,-5,0,0,5,-3,0,0,3,-5,-5,-3,-3,3,&
     3,5,5,-5,-5,-1,-1,1,1,5,5,-3,-3,-1,-1,1,1,3,3,-5,-5,-3,-3,3,3,5,&
     5,-5,-5,-2,-2,2,2,5,5,-3,-3,-2,-2,2,2,3,3,-5,-5,-3,-3,3,3,5,5,-3,&
     -3,3,3/)
 nnz = (/0,0,0,1,0,0,1,1,1,1,1,1,1,1,0,0,2,0,0,0,0,1,1,1,1,2,2,2,&
     2,1,1,1,1,1,1,1,1,2,2,2,2,0,0,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,2,2,&
     2,2,2,0,0,3,0,0,0,0,1,1,1,1,3,3,3,3,1,1,1,1,1,1,1,1,3,3,3,3,0,0,&
     0,0,2,2,2,2,3,3,3,3,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,&
     3,3,2,2,2,2,2,2,2,2,3,3,3,3,0,0,3,3,3,3,1,1,1,1,3,3,3,3,3,3,3,3,&
     2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,0,0,4,0,0,0,0,1,1,1,1,4,4,4,4,1,&
     1,1,1,1,1,1,1,4,4,4,4,0,0,0,0,2,2,2,2,4,4,4,4,1,1,1,1,1,1,1,1,2,&
     2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,4,4,4,4,0,0,0,0,3,&
     3,3,3,4,4,4,4,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,2,&
     2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,4,&
     4,4,4,0,0,4,4,4,4,1,1,1,1,4,4,4,4,4,4,4,4,2,2,2,2,4,4,4,4,4,4,4,&
     4,3,3,3,3,4,4,4,4,4,4,4,4,0,0,5,0,0,0,0,1,1,1,1,5,5,5,5,1,1,1,1,&
     1,1,1,1,5,5,5,5,0,0,0,0,2,2,2,2,5,5,5,5,1,1,1,1,1,1,1,1,2,2,2,2,&
     2,2,2,2,5,5,5,5,5,5,5,5,2,2,2,2,2,2,2,2,5,5,5,5,0,0,0,0,3,3,3,3,&
     5,5,5,5,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,2,2,2,2,&
     2,2,2,2,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,3,5,5,5,5/)
end subroutine get_nnxyz_arrays
 

 
end subroutine link_pairs_by_linked_cells


!subroutine link2_pairs
!! build two lists (for multiple timesteps)
!use sim_cel_data
!use boundaries
!use non_bonded_lists_data
!use paralel_env_data
!use ALL_atoms_data, only : Natoms,xxx,yyy,zzz,l_WALL, xx, yy, zz
!use connectivity_ALL_data, only : list_excluded,size_list_excluded
!use cut_off_data
!use max_sizes_data, only : MX_box,MX_dim,MX_dim3
!real(8) local_cut_short,local_cut_short_sq,local_cut,local_cut_sq
!integer i,j,k
!integer ii_xx,ii_yy,ii_zz
!integer icount
!real(8) r_dim_box_max,r_dim_box_min,rdim,rcheck
!real(8) size_xx,size_yy,size_zz
!integer i_ratio_xx,i_ratio_yy,i_ratio_zz
!integer maxdim
!
!call initialize
!
!CONTAINS
!
!subroutine initialize
!real(8) box_size
! 
!  maxdim = MX_box
!  local_cut_short = cut_off_short + displacement
!  local_cut_short_sq = local_cut_short*local_cut_short
!  local_cut = cut_off + displacement
!  local_cut_sq = local_cut*local_cut
!
!  r_dim_box_max = local_cut/2.0d0 
! 
!!--- OX
!  box_size = cel_cos_a_bxc
!  r_dim_box_min = box_size/natoms**(1.0d0/3.0d0)
!  if (r_dim_box_max < r_dim_box_min) then
!         rdim = r_dim_box_max
!  else
!        icount = 2
!        rcheck = local_cut / dble(icount)
!        do while (rcheck > r_dim_box_min) 
!          icount = icount + 1
!          rcheck = local_cut / dble(icount)
!        enddo
!        rdim = local_cut/dble(icount-1)
!   end if
!   ii_xx =  box_size/rdim
!print*,'rdim ii_xx =',rdim,ii_xx
!   if (ii_xx > maxdim) then
!         write(6,*)' idim > maxdim ',ii_xx,maxdim,' reassigning idim to maxdim '
!         ii_xx = maxdim
!   end if
!   rdim = box_size/ii_xx
!print*,'rdim ii_xx =',rdim,ii_xx
!   i_ratio_xx = local_cut/rdim + 1
!   if (ii_xx .lt. 2*i_ratio_xx+1) then
!         if (mod(ii_xx,2) .eq. 0) then
!           ii_xx = ii_xx + 1
!           rdim = box_size/dble(ii_xx)
!         endif
!         i_ratio_xx = (ii_xx-1)/2
!    end if
!    size_xx = box_size/dble(ii_xx)
!print*, 'ii_xx=',ii_xx, 'i_ratio_xx=',i_ratio_xx, 'size_xx=',size_xx
!print*, 'cel_cos_a_bxc/local_cut xx=', cel_cos_a_bxc/local_cut
!!-- OY
!  box_size = cel_cos_b_axc
!  r_dim_box_min = box_size/natoms**(1.0d0/3.0d0)
!  if (r_dim_box_max < r_dim_box_min) then
!         rdim = r_dim_box_max
!  else
!        icount = 2
!        rcheck = local_cut / dble(icount)
!        do while (rcheck > r_dim_box_min)
!          icount = icount + 1
!          rcheck = local_cut / dble(icount)
!        enddo
!        rdim = local_cut/dble(icount-1)
!   end if
!   ii_yy =  box_size/rdim
!print*,'rdim ii_yy =',rdim,ii_yy
!   if (ii_yy > maxdim) then
!         write(6,*)' idim > maxdim ',ii_yy,maxdim,' reassigning idim to maxdim '
!         ii_yy = maxdim
!   end if
!   rdim = box_size/ii_yy
!print*,'rdim ii_yy =',rdim,ii_yy
!   i_ratio_yy = local_cut/rdim + 1
!   if (ii_yy .lt. 2*i_ratio_yy+1) then
!         if (mod(ii_yy,2) .eq. 0) then
!           ii_yy = ii_yy + 1
!           rdim = box_size/dble(ii_yy)
!         endif
!         i_ratio_yy = (ii_yy-1)/2
!    end if
!    size_yy = box_size/dble(ii_yy)
!print*, 'ii_yy=',ii_yy, 'i_ratio_yy=',i_ratio_yy, 'size_yy=',size_yy
!print*, 'cel_cos_b_axc/local_cut zz=', cel_cos_b_axc/local_cut
!! OZ
!  box_size = cel_cos_c_axb
!  r_dim_box_min = box_size/natoms**(1.0d0/3.0d0)
!  if (r_dim_box_max < r_dim_box_min) then
!         rdim = r_dim_box_max
!  else
!        icount = 2
!        rcheck = local_cut / dble(icount)
!        do while (rcheck > r_dim_box_min)
!          icount = icount + 1
!          rcheck = local_cut / dble(icount)
!        enddo
!        rdim = local_cut/dble(icount-1)
!   end if
!   ii_zz =  box_size/rdim
!print*,'rdim ii_zz =',rdim,ii_zz
!   if (ii_zz > maxdim) then
!         write(6,*)' idim > maxdim ',ii_zz,maxdim , ' reassigning idim to maxdim '
!         ii_zz = maxdim
!   end if
!   rdim = box_size/ii_zz
!print*,'rdim ii_zz =',rdim,ii_zz
!   i_ratio_zz = local_cut/rdim + 1
!   if (ii_zz .lt. 2*i_ratio_zz+1) then
!         if (mod(ii_zz,2) .eq. 0) then
!           ii_zz = ii_zz + 1
!           rdim = box_size/dble(ii_zz)
!         endif
!         i_ratio_zz = (ii_zz-1)/2
!    end if
!    size_zz = box_size/dble(ii_zz)
!print*, 'ii_zz=',ii_zz, 'i_ratio_zz=',i_ratio_zz, 'size_zz=',size_zz
!print*, 'cel_cos_c_axb/local_cut zz=', cel_cos_c_axb/local_cut
!    
!end subroutine initialize
!end subroutine link2_pairs
!     

end module  non_bonded_lists_builder
