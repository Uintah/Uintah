 module excuded_list_eval_module
  implicit none
  public :: get_excluded_lists

  contains 
 subroutine get_excluded_lists
 use connectivity_ALL_data
 use atom_type_data
 use ALL_atoms_data, only : i_type_atom,atom_in_which_molecule, is_dummy,is_sfield_constrained
 use ALL_mols_data
 use paralel_env_data
 use integrate_data, only : l_do_QN_CTRL
  implicit none
  integer, allocatable :: next_site(:)
  integer i,j,k,ia,ib,ic,id,i1,j1,k1,ineigh,i14a,i14b,i_bond_1,i_bond_2,ibond
  integer i_angle_1,i_angle_2,i_angle_3,iangle,itmp
  logical l_skip, HasBeenUsed,Found14
  integer, allocatable :: temp_size_list_14(:),temp_list_14(:,:)
  allocate(next_site(Natoms))
  next_site = 0
  list_excluded = 0

! do first the rigid bodies
  do i = 1, Nmols
   if (l_RIGID_GROUP(i)) then
     do i1 = start_group(i),end_group(i)-1
     do j1 = i1 + 1, end_group(i)
       l_skip = .false.
       do j = 1, min(next_site(i1),MX_excluded)
        if (list_excluded(i1,j) == j1) then
          l_skip = .true.
        endif
       enddo
       if (.not.l_skip) then
         next_site(i1) = next_site(i1) + 1
         next_site(j1) = next_site(j1) + 1
         if (max(next_site(i1),next_site(j1)) > MX_excluded) then
           print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
           print*, ' next_site(i1) ; next_site(j1) ; MX_excluded =',next_site(i1),next_site(j1),MX_excluded
           STOP
         endif
         list_excluded(i1, next_site(i1)) = j1
         list_excluded(j1, next_site(j1)) = i1
       endif ! (.not.l_skip)
     enddo
     enddo
   endif
  enddo


if (.not.l_do_QN_CTRL) then
!!!!!!!
  do i = 1, Nbonds
    ia = list_bonds(1,i)
    ib = list_bonds(2,i)
    l_skip = .false.
    do j = 1, min(max(next_site(ia),next_site(ib)),MX_excluded)
      if (list_excluded(ia,j) == ib) then
        l_skip = .true.
        goto 1
      endif
    enddo
1 continue
    if (.not.l_skip) then
      next_site(ia) = next_site(ia) + 1
      next_site(ib) = next_site(ib) + 1
      if (max(next_site(ia),next_site(ib)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ia) ; next_site(ib) ; MX_excluded =',next_site(ia),next_site(ib),MX_excluded
        STOP
      endif
      list_excluded(ia, next_site(ia)) = ib
      list_excluded(ib, next_site(ib)) = ia
    endif
  enddo


  do i = 1, Nconstrains
    ia = list_constrains(1,i)
    ib = list_constrains(2,i)
    l_skip = .false.
    do j = 1, min(max(next_site(ia),next_site(ib)),MX_excluded)
      if (list_excluded(ia,j) == ib) then
        l_skip = .true.
        goto 2
      endif
    enddo
2 continue
    if (.not.l_skip) then
      next_site(ia) = next_site(ia) + 1
      next_site(ib) = next_site(ib) + 1
      if (max(next_site(ia),next_site(ib)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ia) ; next_site(ib) ; MX_excluded =',next_site(ia),next_site(ib),MX_excluded
        STOP
      endif
      list_excluded(ia, next_site(ia)) = ib
      list_excluded(ib, next_site(ib)) = ia
    endif
  enddo ! Nconstrains
! do anglea

    do i = 1, Nangles
    ia = list_angles(1,i)
    ib = list_angles(2,i)
    ic = list_angles(3,i)
if (ia==0.or.ib==0.or.ic==0)then
print*,i,'CLUSTER F**K!!!',ia,ib,ic
stop
endif
! ia - ib
    l_skip = .false.
    do j = 1, min(max(next_site(ia),next_site(ib)),MX_excluded)
      if (list_excluded(ia,j) == ib) then
        l_skip = .true.
        goto 31
      endif
    enddo
31 continue
    if (.not.l_skip) then
      next_site(ia) = next_site(ia) + 1
      next_site(ib) = next_site(ib) + 1
      if (max(next_site(ia),next_site(ib)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ia) ; next_site(ib) ; MX_excluded =',next_site(ia),next_site(ib),MX_excluded
        STOP
      endif
      list_excluded(ia, next_site(ia)) = ib
      list_excluded(ib, next_site(ib)) = ia

    endif

! ib - ic
      l_skip = .false.
    do j = 1, min(max(next_site(ib),next_site(ic)),MX_excluded)
      if (list_excluded(ib,j) == ic) then
        l_skip = .true.
        goto 32
      endif
    enddo
32 continue
    if (.not.l_skip) then
      next_site(ib) = next_site(ib) + 1
      next_site(ic) = next_site(ic) + 1
      if (max(next_site(ib),next_site(ic)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ib) ; next_site(ic) ; MX_excluded =',next_site(ib),next_site(ic),MX_excluded
        STOP
      endif
      list_excluded(ib, next_site(ib)) = ic
      list_excluded(ic, next_site(ic)) = ib
    endif

! ia - ic
         l_skip = .false.
    do j = 1, min(max(next_site(ia),next_site(ic)),MX_excluded)
      if (list_excluded(ia,j) == ic) then
        l_skip = .true.
        goto 33
      endif
    enddo
33 continue
    if (.not.l_skip) then
      next_site(ia) = next_site(ia) + 1
      next_site(ic) = next_site(ic) + 1
      if (max(next_site(ia),next_site(ic)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ia) ; next_site(ic) ; MX_excluded =',next_site(ia),next_site(ic),MX_excluded
        STOP
      endif
      list_excluded(ia, next_site(ia)) = ic
      list_excluded(ic, next_site(ic)) = ia
    endif

  enddo ! Nangles
! dihedrals



   if (l_exclude_14_dih_CTRL) then
    do i = 1, Ndihedrals
    ia = list_dihedrals(1,i)
    ib = list_dihedrals(2,i)
    ic = list_dihedrals(3,i)
    id = list_dihedrals(4,i)
! ia - ib

    l_skip = .false.
    do j = 1, min(max(next_site(ia),next_site(ib)),MX_excluded)
      if (list_excluded(ia,j) == ib) then
        l_skip = .true.
        goto 41
      endif
    enddo
41 continue
    if (.not.l_skip) then
      next_site(ia) = next_site(ia) + 1
      next_site(ib) = next_site(ib) + 1
      if (max(next_site(ia),next_site(ib)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ia) ; next_site(ib) ; MX_excluded =',next_site(ia),next_site(ib),MX_excluded
        STOP
      endif
      list_excluded(ia, next_site(ia)) = ib
      list_excluded(ib, next_site(ib)) = ia
    endif
! ib - ic
      l_skip = .false.
    do j = 1, min(max(next_site(ib),next_site(ic)),MX_excluded)
      if (list_excluded(ib,j) == ic) then
        l_skip = .true.
        goto 42
42 continue
      endif
    enddo
    if (.not.l_skip) then
      next_site(ib) = next_site(ib) + 1
      next_site(ic) = next_site(ic) + 1
      if (max(next_site(ib),next_site(ic)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ib) ; next_site(ic) ; MX_excluded =',next_site(ib),next_site(ic),MX_excluded
        STOP
      endif
      list_excluded(ib, next_site(ib)) = ic
      list_excluded(ic, next_site(ic)) = ib
    endif

! ia - ic
         l_skip = .false.
    do j = 1, min(max(next_site(ia),next_site(ic)),MX_excluded)
      if (list_excluded(ia,j) == ic) then
        l_skip = .true.
      goto 43
      endif
    enddo
43 continue
    if (.not.l_skip) then
      next_site(ia) = next_site(ia) + 1
      next_site(ic) = next_site(ic) + 1
      if (max(next_site(ia),next_site(ic)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ia) ; next_site(ic) ; MX_excluded =',next_site(ia),next_site(ic),MX_excluded
        STOP
      endif
      list_excluded(ia, next_site(ia)) = ic
      list_excluded(ic, next_site(ic)) = ia
    endif

! ia - id
         l_skip = .false.
    do j = 1, min(max(next_site(ia),next_site(id)),MX_excluded)
      if (list_excluded(ia,j) == id) then
        l_skip = .true.
        goto 44
      endif
    enddo
44 continue
    if (.not.l_skip) then
      next_site(ia) = next_site(ia) + 1
      next_site(id) = next_site(id) + 1
      if (max(next_site(ia),next_site(id)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ia) ; next_site(id) ; MX_excluded =',next_site(ia),next_site(id),MX_excluded
        STOP
      endif
      list_excluded(ia, next_site(ia)) = id
      list_excluded(id, next_site(id)) = ia
    endif

! ib - id
         l_skip = .false.
    do j = 1, min(max(next_site(ib),next_site(id)),MX_excluded)
      if (list_excluded(ib,j) == id) then
        l_skip = .true.
        goto 45
      endif
    enddo
45 continue
    if (.not.l_skip) then
      next_site(ib) = next_site(ib) + 1
      next_site(id) = next_site(id) + 1
      if (max(next_site(ib),next_site(id)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ib) ; next_site(id) ; MX_excluded =',next_site(ib),next_site(id),MX_excluded
        STOP
      endif
      list_excluded(ib, next_site(ib)) = id
      list_excluded(id, next_site(id)) = ib
    endif
! ic - id
         l_skip = .false.
    do j = 1, min(max(next_site(ic),next_site(id)),MX_excluded)
      if (list_excluded(ic,j) == id) then
        l_skip = .true.
        goto 46
      endif
    enddo
46 continue
    if (.not.l_skip) then
      next_site(ic) = next_site(ic) + 1
      next_site(id) = next_site(id) + 1
      if (max(next_site(ic),next_site(id)) > MX_excluded) then
        print*, 'ERROR in get_excluded_lists: Increase the size of MX_excluded'
        print*, ' next_site(ic) ; next_site(id) ; MX_excluded =',next_site(ic),next_site(id),MX_excluded
        STOP
      endif
      list_excluded(ic, next_site(ic)) = id
      list_excluded(id, next_site(id)) = ic
    endif

  enddo ! Ndihedrals

 endif !(l_exclude_14_dih_CTRL)

ENDIF ! l_do_QN_CTRL

 size_list_excluded = next_site


 size_list_excluded_HALF=0.0d0; list_excluded_HALF=0.0d0
 do i = 1, Natoms-1
    do k = 1, size_list_excluded(i)
      j = list_excluded(i,k)
      if (i < j) then
        size_list_excluded_HALF(i) = size_list_excluded_HALF(i)+1
        list_excluded_HALF(i,size_list_excluded_HALF(i)) = j
      endif
    enddo
 enddo !

if(.not.l_do_QN_CTRL) then
! DO the 1-4 list
i14a=0
i14b=0
     if (l_build_14_from_angle_CTRL) then
         if (l_red_14_vdw_CTRL.or.l_red_14_Q_mu_CTRL.or.l_red_14_Q_CTRL.or.l_red_14_mu_mu_CTRL) then
           do iangle=1,Nangles
             i_angle_1=list_angles(1,iangle)
             i_angle_2=list_angles(2,iangle)
             i_angle_3=list_angles(3,iangle)
             do ibond=1,Nbonds
               Found14=.false.
               i_bond_1=list_bonds(1,ibond)
               i_bond_2=list_bonds(2,ibond)
               if (i_bond_1==i_angle_3.and.i_bond_2/=i_angle_2) then
                 i14a=i_angle_1
                 i14b=i_bond_2
                 Found14=.true.
               end if
               if (i_bond_2==i_angle_3.and.i_bond_1/=i_angle_2) then
                 i14a=i_angle_1
                 i14b=i_bond_1
                 Found14=.true.
               end if
               if (i_bond_1==i_angle_1.and.i_bond_2/=i_angle_2) then
                 i14a=i_angle_3
                 i14b=i_bond_2
                 Found14=.true.
               end if
               if (i_bond_2==i_angle_1.and.i_bond_1/=i_angle_2) then
                 i14a=i_angle_3
                 i14b=i_bond_1
                 Found14=.true.
               end if
               if (Found14) then
                 if (is_dummy(i14a).or.is_dummy(i14b)) Found14=.false.
               end if

              if (i14a==i14b) Found14=.false.
               if (Found14) then
                 if (i14a.gt.i14b) then
                   itmp=i14a
                   i14a=i14b
                   i14b=itmp
                 end if
                 HasBeenUsed=.false.
                 do ineigh = 1,size_list_14(i14a)
                   if (list_14(i14a,ineigh)==i14b) HasBeenUsed=.true.
                 end do
                 do ineigh = 1,size_list_14(i14b)
                   if (list_14(i14b,ineigh)==i14a) HasBeenUsed=.true.
                 end do
!C
!C  remove 1-4 that are part of listex (matters for cyclic molecules) (v2.7b)
!C
                 do i=1,size_list_excluded_HALF(i14a)
                   if (list_excluded_HALF(i14a,i)==i14b) HasBeenUsed=.true.
                 end do
!C
                 if (.not.HasBeenUsed) then
                   size_list_14(i14a)=size_list_14(i14a)+1
                   if (size_list_14(i14a)>MX_in_list_14) then
                       print*, 'ERROR: when build the list 14; size_list_14(i14a)>MX_in_list_14 ',&
                       size_list_14(i14a),MX_in_list_14
                       print*, 'Increase the parameter : MX_in_list_14 in header_list.f90 and recompile and restart'
                       STOP
                   endif
                   list_14(i14a, size_list_14(i14a))=i14b
                 end if
               end if
             end do
          end do
        endif ! (lredonefour)
      endif ! l_build_14_from_angle_CTRL
      
      N_pairs_14 = 0 ; N_pairs_14 = sum(size_list_14)
print*, 'N_pairs_14=',N_pairs_14
ENDIF ! l_do_QN_CTRL

      if (l_do_QN_CTRL) N_pairs_14 = 0 ! if rigid and quaternions then no interaction 14. 


 include 'finalize_list_exclude.frg'

 ! carefull to update lists in DoDummyInit
 
 end subroutine get_excluded_lists

 end module excuded_list_eval_module
