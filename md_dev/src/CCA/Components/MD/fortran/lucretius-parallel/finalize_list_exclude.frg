do i = 1, Natoms ! I add 12 and 13 interactions to
  k1 = 0
  do k = 1, size_list_excluded_HALF(i)
    j = list_excluded_HALF(i,k)
    if (is_sfield_constrained(i).and.is_sfield_constrained(j)) then
      k1 = k1 + 1
      list_excluded_sfc_iANDj_HALF(i,k1) = j
    endif
  enddo
  size_list_excluded_sfc_iANDj_HALF(i) = k1
enddo

N_pairs_sfc_sfc_123= sum(size_list_excluded_sfc_iANDj_HALF)

if (N_pairs_sfc_sfc_123>0)then
  allocate(temp_size_list_14(Natoms))
  allocate(temp_list_14(Natoms, lbound(list_14,dim=2):ubound(list_14,dim=2)));
  temp_list_14=0;temp_size_list_14=0

  do i = 1, Natoms
  k1=0
  do k = 1, size_list_14(i)
   j=list_14(i,k)
   if (.not.(is_sfield_constrained(i).and.is_sfield_constrained(j))) then
     k1 = k1 + 1
     temp_list_14(i,k1) = j
   endif
  enddo
  temp_size_list_14(i) = k1
  enddo ! i
  size_list_14 = temp_size_list_14
  list_14 = temp_list_14
  N_pairs_14 = 0 ;  N_pairs_14 = sum(size_list_14)
  deallocate(temp_size_list_14,temp_list_14)

endif


  do i = 1, Natoms
  k1=0
  do k = 1, size_list_excluded_HALF(i)
   j=list_excluded_HALF(i,k)
   if (.not.(is_sfield_constrained(i).and.is_sfield_constrained(j))) then
     k1 = k1 + 1
     list_excluded_HALF_no_SFC(i,k1) = j
   endif
  enddo
  size_list_excluded_HALF_no_SFC(i) = k1
  enddo ! i

! UPDATE whole list excluded.
 list_excluded=0
 size_list_excluded=0
 do i = 1, Natoms
  do k = 1, size_list_excluded_HALF(i)
    j = list_excluded_HALF(i,k)
    size_list_excluded(i)=size_list_excluded(i)+1
    list_excluded(i,size_list_excluded(i)) = j
    size_list_excluded(j)=size_list_excluded(j)+1
    list_excluded(j,size_list_excluded(j)) = i
  enddo
 enddo
! \finally validate excluded list

 do i = 1, Natoms
   do j = 1, size_list_excluded(i)
     if (list_excluded(i,j) <= 0 .or. list_excluded(i,j)> Natoms) then
     print*, 'CLUSTERF*CK IN get_excluded_lists; list_excluded(i,j) not within the defined range [0..Natoms]',i,j,&
      'list_excluded(i,j)=',list_excluded(i,j)
     STOP
     endif
     do k = j+1,size_list_excluded(i)
        if (list_excluded(i,j)==list_excluded(i,k)) then
          print*, 'CLUSTERF*CK IN get_excluded_lists; 2 list_excluded are equal for atom ',i, ' jk=',j,k,list_excluded(i,j),list_excluded(i,k)
          STOP
        endif
     enddo
    enddo
 enddo
 deallocate(next_site)

 do i = 1, Natoms
   do j = 1, size_list_14(i)
    if (list_14(i,j) <= 0 .or. list_14(i,j) > Natoms) then
       print*, 'CLUSTERF*CK IN building list 1-4; list_14(i,j) not within the defined range [0..Natoms]',i,j,&
      'list_excluded(i,j)=',list_14(i,j)
       STOP
    endif
    enddo
    do j = 1, size_list_14(i)
    do k = j + 1, size_list_14(i)
       if (list_14(i,j) == list_14(i,k) ) then
        print*, 'CLUSTERF*CK IN building list 1-4; list_14(i,j) 2 list_14 are equal for atom ',i,' j k =',j,k,list_14(i,j),list_14(i,k)
        STOP
       endif
    enddo
    enddo
 enddo

