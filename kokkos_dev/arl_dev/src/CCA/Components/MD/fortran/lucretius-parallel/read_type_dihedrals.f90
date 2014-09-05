 subroutine read_type_dihedrals(imol, i1, dihedral_2_mol, &
            i_start, i_end, NumberOfWords, Max_words_per_line, the_words, &
                             where_in_file, nf)
 use atom_type_data
 use mol_type_data
 use max_sizes_data, only : Max_Atom_Name_Len
 use file_names_data, only :  MAX_CH_size
 use force_field_data, only : predef_ff_dih, N_predef_ff_dihs
 use CTRLs_data
 use types_module, only : word_type
 use field_constrain_data, only : is_type_atom_field_constrained
 use chars, only : locate_word_in_key,attempt_integer_strict,attempt_real_strict,UP_CASE,&
                   select_int_real_text_type
 use connectivity_type_data
 use intramol_forces_def

 implicit none
 integer, intent(IN) :: imol
 integer, intent( inout) :: i1 ! 
 integer, intent(IN) :: dihedral_2_mol(:)
 integer, intent(in) :: i_start, i_end, Max_words_per_line
 type(word_type),intent(in) :: the_words(i_start:i_end,1:Max_words_per_line)
 integer, intent(IN) :: NumberOfWords(i_start:i_end), where_in_file(i_start:i_end)
 character(*) , intent(IN) :: nf
 character(MAX_CH_size) chtemp

 logical l_OK, l_found, is_constrain, l_any, l_ID
 integer i,j,k,kkk, i10, iii, NNN, jjj, ijk,Nprms, ibla, i_type, int_1, int_2, iii1,iii2, i_index

   do j = i_start, i_end
   if (dihedral_2_mol(imol) > 0) then

     i1 = i1 + 1
     if (NumberOfWords(j) < 6) then
        write(6,*) 'ERROR in the file"',trim(nf),'" at the line ',where_in_file(j), &
                  'there must be at least 5 records and you only have ',&
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
                                    dihedral_types(2,i1) , trim(nf), where_in_file(j))

     kkk = 3
     NNN = the_words(j,kkk)%length
     call attempt_integer_strict(NNN,the_words(j,kkk)%ch(1:NNN), &
                                     dihedral_types(3,i1) , trim(nf), where_in_file(j))

     kkk = 4
     NNN = the_words(j,kkk)%length
     call attempt_integer_strict(NNN,the_words(j,kkk)%ch(1:NNN), &
                                     dihedral_types(4,i1) , trim(nf), where_in_file(j))

     kkk = 5
     NNN = the_words(j,kkk)%length
     call attempt_integer_strict(NNN,the_words(j,kkk)%ch(1:NNN), &
                                     dihedral_types(5,i1) , trim(nf), where_in_file(j))

     do kkk = 2,5
     do iii1 = kkk+1,5
        if (dihedral_types(kkk,i1)==dihedral_types(iii1,i1) ) then
          print*, 'ERROR in file',trim(nf),' at line', where_in_file(j), ' entires ',kkk,iii1,&
          ' are equal and are:',dihedral_types(kkk,i1),dihedral_types(iii1,i1), ' They must be diferent '
         STOP
        endif
     enddo
     enddo


! see if constrain
                        
     NNN = the_words(j,6)%length
     call select_int_real_text_type(NNN,the_words(j,6)%ch(1:NNN), i_type)
     l_ID = .false.
     if (i_type==1) then
           call attempt_integer_strict(NNN,the_words(j,6)%ch(1:NNN), &
                                        i_index  , trim(nf), where_in_file(j))
           if (N_predef_ff_dihs > 0) then
           if (i_index >= lbound(predef_ff_dih,dim=1) .and. i_index <= ubound(predef_ff_dih,dim=1)) then
           l_ID = .true.
              dihedral_types(1,i1) = predef_ff_dih(i_index)%style
              prm_dihedral_types(1:MX_DIH_PRMS,i1) = predef_ff_dih(i_index)%the_params(1:MX_DIH_PRMS)
              Nfolds_dihedral_types(i1) = predef_ff_dih(i_index)%N_params
          else   ! (i_index > lbound(predef_ff_dihedral,dim=1)-1 .and. i_i
            print*, 'ERROR in file ',trim(nf), ' the record from line ',&
             where_in_file(j), ' coloumn 6 must be in the range', &
             lbound(predef_ff_dih,dim=1),ubound(predef_ff_dih,dim=1), &
             ' instead of its value :',i_index
             STOP
          endif  ! (i_index > lbound(predef_ff_dihedral,dim=1)-1 .and. i_i
          endif  !  (N_predef_ff_dihedrals > 0) then
      else if (i_type==3) then
         do iii2 = 1, N_predef_ff_dihs
          if (trim(predef_ff_dih(iii2)%name)==trim(the_words(j,6)%ch(1:NNN))) then
          l_ID = .true.
              dihedral_types(1,i1) = predef_ff_dih(iii2)%style
              prm_dihedral_types(1:MX_DIH_PRMS,i1) = predef_ff_dih(iii2)%the_params(1:MX_DIH_PRMS)
              Nfolds_dihedral_types(i1) = predef_ff_dih(iii2)%N_params
          goto 33
          endif ! (trim(predef_ff_dihedral(iii2)%name)==trim(the_words(j,4)%ch(1:NNN)))
        enddo  ! do iii2 = 1, N_predef_ff_dihedrals

33 continue
        
      else  ! i_type ==2 
         print*, 'ERROR in file', trim(nf), ' a real number was not expected at line', where_in_file(j), &
         ' but expected is an integer or a text defining the dihedrals'
         STOP
      endif

      
      if (.not.l_ID) then ! the dihedral is defined here and not in ff
            NNN = the_words(j,6)%length
            if (.not.is_dihedral(trim(the_words(j,6)%ch(1:NNN)))) then
               print*, 'ERROR in file ', trim(nf), ' at line ', where_in_file(j), ' NO DIHEDRAL was defined at 6th coloumn'
               STOP
            endif
            call get_style_dihedral(trim(the_words(j,6)%ch(1:NNN)),Nprms,dihedral_types(1,i1))
            call locate_word_in_key(trim(the_words(j,6)%ch(1:NNN)),Nprms,the_words(j,:), l_found,kkk) ! for validation
            if (dihedral_types(1,i1) == 1) then
              iii1 = 2
              iii2 = Nprms + 1
              call attempt_integer_strict(NNN,the_words(j,kkk+ijk)%ch(1:NNN), &
                                        Nfolds_dihedral_types(i1)  , trim(nf), where_in_file(j))
            else
              iii1 = 1
              iii2 = Nprms 
            endif
            do ijk = iii1,iii2
               NNN = the_words(j,kkk+ijk)%length
               call attempt_real_strict(NNN,the_words(j,kkk+ijk)%ch(1:NNN), &
                                        prm_dihedral_types(ijk-iii1+1,i1)  , trim(nf), where_in_file(j))
            enddo
      endif ! not.l_ID




endif!
enddo ! i


end subroutine read_type_dihedrals

!\see if constrain
             

