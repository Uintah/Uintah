 subroutine read_type_bonds(imol, i1,i_bond, i_constrain, bond_2_mol, &
            i_start, i_end, NumberOfWords, Max_words_per_line, the_words, &
                             where_in_file, nf)
 use atom_type_data
 use mol_type_data
 use max_sizes_data, only : Max_Atom_Name_Len
 use file_names_data, only :  MAX_CH_size
 use force_field_data, only : predef_ff_bond, N_predef_ff_bonds
 use CTRLs_data
 use types_module, only : word_type
 use field_constrain_data, only : is_type_atom_field_constrained
 use chars, only : locate_word_in_key,attempt_integer_strict,attempt_real_strict,UP_CASE,&
                   select_int_real_text_type
 use connectivity_type_data
 use intramol_forces_def

 implicit none
 integer, intent(IN) :: imol
 integer, intent( inout) :: i1,i_bond, i_constrain ! i1 is the actual type of the atom
 integer, intent(IN) :: bond_2_mol(:)
 integer, intent(in) :: i_start, i_end, Max_words_per_line
 type(word_type),intent(in) :: the_words(i_start:i_end,1:Max_words_per_line)
 integer, intent(IN) :: NumberOfWords(i_start:i_end), where_in_file(i_start:i_end)
 character(*) , intent(IN) :: nf
 character(MAX_CH_size) chtemp

 logical l_OK, l_found, is_constrain, l_any, l_ID, l_ID1
 integer i,j,k,kkk, i10, iii, NNN, jjj, ijk,Nprms, ibla, i_type, int_1, int_2, iii1,iii2, i_index
do j = i_start, i_end
if (bond_2_mol(imol) > 0) then

     i1 = i1 + 1
     if (NumberOfWords(j) < 4) then
        write(6,*) 'ERROR in the file"',trim(nf),'" at the line ',where_in_file(j), &
                  'there must be at least 4 records and you only have ',&
                   NumberOfWords(j)
        STOP
     endif
     kkk = 1
     NNN = the_words(j,kkk)%length
     call attempt_integer_strict(NNN,the_words(j,kkk)%ch(1:NNN), &
                                        ibla  , trim(nf), where_in_file(j))
     kkk = 2
     NNN = the_words(j,kkk)%length
     call attempt_integer_strict(NNN,the_words(j,kkk)%ch(1:NNN), &
                                        int_1  , trim(nf), where_in_file(j))

     kkk = 3
     NNN = the_words(j,kkk)%length
     call attempt_integer_strict(NNN,the_words(j,kkk)%ch(1:NNN), &
                                        int_2  , trim(nf), where_in_file(j))

     if (int_1 == int_2) then
          print*, 'ERROR in file',trim(nf),' at line', where_in_file(j), ' entires 2 and 3 ',&
          ' are equal and are:',int_1,int_2, ' They must be diferent '
         STOP
     endif
! see if constrain
                        
     NNN = the_words(j,4)%length
     call select_int_real_text_type(NNN,the_words(j,4)%ch(1:NNN), i_type)
     l_ID = .false.
     is_constrain = .false.
     if (i_type==1) then
           call attempt_integer_strict(NNN,the_words(j,4)%ch(1:NNN), &
                                        i_index  , trim(nf), where_in_file(j))
           if (N_predef_ff_bonds > 0) then
           if (i_index >= lbound(predef_ff_bond,dim=1) .and. i_index <= ubound(predef_ff_bond,dim=1)) then
           l_ID = .true.
            i_bond = i_bond + 1
            bond_types(1,i_bond) = predef_ff_bond(i_index)%style
            bond_types(2,i_bond) = int_1
            bond_types(3,i_bond) = int_2
            prm_bond_types(1:MX_BOND_PRMS,i_bond) = predef_ff_bond(i_index)%the_params(1:MX_BOND_PRMS)
!print*,'i_index predef_ff_bond(i_index)%is_constrain=',i_index,predef_ff_bond(i_index)%is_constrain
            l_ID1 = .false.
            if (predef_ff_bond(i_index)%is_constrain) then
               l_ID1 = .true.
               i_constrain = i_constrain + 1
               prm_constrain_types(1,i_constrain) = predef_ff_bond(i_index)%constrained_bond
               is_constrain = .true.
               constrain_types(1,i_constrain) = int_1
               constrain_types(2,i_constrain) = int_2
               is_type_bond_constrained(i_bond) = .true.
            endif
          else   ! (i_index > lbound(predef_ff_bond,dim=1)-1 .and. i_i
            print*, 'ERROR in file ',trim(nf), ' the record from line ',&
             where_in_file(j), ' coloumn 4 must be in the range', &
             lbound(predef_ff_bond,dim=1),ubound(predef_ff_bond,dim=1), &
             ' instead of its value :',i_index
             STOP
          endif  ! (i_index > lbound(predef_ff_bond,dim=1)-1 .and. i_i
          endif  !  (N_predef_ff_bonds > 0) then
      else if (i_type==3) then
         do iii2 = 1, N_predef_ff_bonds
          if (trim(predef_ff_bond(iii2)%name)==trim(the_words(j,4)%ch(1:NNN))) then
          l_ID = .true.
          l_ID1=.false.
              i_bond = i_bond + 1
              bond_types(1,i_bond) = predef_ff_bond(iii2)%style
              bond_types(2,i_bond) = int_1
              bond_types(3,i_bond) = int_2
              prm_bond_types(1:MX_BOND_PRMS,i_bond) = predef_ff_bond(iii2)%the_params(1:MX_BOND_PRMS)
              if (predef_ff_bond(iii2)%is_constrain) then
               l_ID1 = .true.
               i_constrain = i_constrain + 1
               prm_constrain_types(1,i_constrain) = predef_ff_bond(iii2)%constrained_bond
               is_constrain = .true.
               constrain_types(1,i_constrain) = int_1
               constrain_types(2,i_constrain) = int_2
               is_type_bond_constrained(i_bond) = .true.
               goto 33
            endif
          endif ! (trim(predef_ff_bond(iii2)%name)==trim(the_words(j,4)%ch(1:NNN)))
        enddo  ! do iii2 = 1, N_predef_ff_bonds

33 continue
        
      else  ! i_type ==2 
         print*, 'ERROR in file', trim(nf), ' a real number was not expected at line', where_in_file(j), &
         ' but an integer or a text defining the bond'
         STOP
      endif

      
          is_constrain=.false.
          do iii1 = 4 , NumberOfWords(j)
          NNN = the_words(j,iii1)%length
           do k = 1, NNN
             chtemp(k:k) = UP_CASE(the_words(j,iii1)%ch(k:k))
           enddo
           if (chtemp(1:NNN) == '*CONS') then
             is_constrain = .true.
             goto 3
          endif
        enddo ! iii4,NumberOfWords(j)
 3 continue
        if (is_constrain) then
           if (l_ID) then
           print*,'WARNING: The bond definition from the force field is redefined as a constraint in the file',trim(nf), &
           'at line ',where_in_file(j)
           endif
           if (.not.l_ID) then
             i_bond = i_bond + 1
             i_constrain = i_constrain + 1
           else
             if (.not.l_ID1) i_constrain = i_constrain + 1
           endif
!print*,'i_bond i_constrain=',i_bond,i_constrain
           bond_types(2,i_bond) = int_1
           bond_types(3,i_bond) = int_2
           constrain_types(1,i_constrain) = int_1
           constrain_types(2,i_constrain) = int_2
           is_type_bond_constrained(i_bond) = .true.
           if (NumberOfWords(j) < iii1 + 1 ) then
               print*, 'ERROR in file',trim(nf), ' more records are needed'
               STOP
            endif
           NNN = the_words(j,iii1+1)%length
           call attempt_real_strict(NNN,the_words(j,iii1+1)%ch(1:NNN), &
                prm_constrain_types(1,i_constrain)  , trim(nf), where_in_file(j))
        else
        if (.not.l_ID) then
            i_bond = i_bond + 1
            is_type_bond_constrained(i_bond) = .false.
            NNN = the_words(j,4)%length
            if (.not.is_bond(trim(the_words(j,4)%ch(1:NNN)))) then
               print*, 'ERROR in file ', trim(nf), ' at line ', where_in_file(j), ' NO BOND was defined at 4th coloumn'
               STOP
            endif
            bond_types(2,i_bond) = int_1
            bond_types(3,i_bond) = int_2
            call get_style_bond(trim(the_words(j,4)%ch(1:NNN)),Nprms,bond_types(1,i_bond))
            call locate_word_in_key(trim(the_words(j,4)%ch(1:NNN)),Nprms,the_words(j,:), l_found,kkk) ! for validation
            do ijk = 1,Nprms
               NNN = the_words(j,kkk+ijk)%length
               call attempt_real_strict(NNN,the_words(j,kkk+ijk)%ch(1:NNN), &
                                        prm_bond_types(ijk,i_bond)  , trim(nf), where_in_file(j))
            enddo
         endif ! .not.l_ID
         end if ! is_constr




endif!
enddo ! i

end subroutine read_type_bonds

!\see if constrain
             

