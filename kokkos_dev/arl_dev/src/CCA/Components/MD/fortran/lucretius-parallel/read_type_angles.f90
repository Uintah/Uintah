 subroutine read_type_angles(imol, i1, angle_2_mol, &
            i_start, i_end, NumberOfWords, Max_words_per_line, the_words, &
                             where_in_file, nf)
 use atom_type_data
 use mol_type_data
 use max_sizes_data, only : Max_Atom_Name_Len
 use file_names_data, only :  MAX_CH_size
 use force_field_data, only : predef_ff_angle, N_predef_ff_angles
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
 integer, intent(IN) :: angle_2_mol(:)
 integer, intent(in) :: i_start, i_end, Max_words_per_line
 type(word_type),intent(in) :: the_words(i_start:i_end,1:Max_words_per_line)
 integer, intent(IN) :: NumberOfWords(i_start:i_end), where_in_file(i_start:i_end)
 character(*) , intent(IN) :: nf
 character(MAX_CH_size) chtemp

 logical l_OK, l_found, is_constrain, l_any, l_ID
 integer i,j,k,kkk, i10, iii, NNN, jjj, ijk,Nprms, ibla, i_type, int_1, int_2, iii1,iii2, i_index

   do j = i_start, i_end
   if (angle_2_mol(imol) > 0) then

     i1 = i1 + 1
     if (NumberOfWords(j) < 5) then
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
                                    angle_types(2,i1) , trim(nf), where_in_file(j))

     kkk = 3
     NNN = the_words(j,kkk)%length
     call attempt_integer_strict(NNN,the_words(j,kkk)%ch(1:NNN), &
                                     angle_types(3,i1) , trim(nf), where_in_file(j))

     kkk = 4
     NNN = the_words(j,kkk)%length
     call attempt_integer_strict(NNN,the_words(j,kkk)%ch(1:NNN), &
                                     angle_types(4,i1) , trim(nf), where_in_file(j))
     do kkk = 2,4
     do iii1 = kkk+1,4
        if (angle_types(kkk,i1)==angle_types(iii1,i1) ) then
          print*, 'ERROR in file',trim(nf),' at line', where_in_file(j), ' entires ',kkk,iii1,&
          ' are equal and are:',angle_types(kkk,i1),angle_types(iii1,i1), ' They must be diferent '
         STOP
        endif
     enddo
     enddo


! see if constrain
                        
     NNN = the_words(j,5)%length
     call select_int_real_text_type(NNN,the_words(j,5)%ch(1:NNN), i_type)
     l_ID = .false.
     if (i_type==1) then
           call attempt_integer_strict(NNN,the_words(j,5)%ch(1:NNN), &
                                        i_index  , trim(nf), where_in_file(j))
           if (N_predef_ff_angles > 0) then
           if (i_index >= lbound(predef_ff_angle,dim=1) .and. i_index <= ubound(predef_ff_angle,dim=1)) then
           l_ID = .true.
              angle_types(1,i1) = predef_ff_angle(i_index)%style
              prm_angle_types(1:MX_ANGLE_PRMS,i1) = predef_ff_angle(i_index)%the_params(1:MX_ANGLE_PRMS)
          else   ! (i_index > lbound(predef_ff_angle,dim=1)-1 .and. i_i
            print*, 'ERROR in file ',trim(nf), ' the record from line ',&
             where_in_file(j), ' coloumn 5 must be in the range', &
             lbound(predef_ff_angle,dim=1),ubound(predef_ff_angle,dim=1), &
             ' instead of its value :',i_index
             STOP
          endif  ! (i_index > lbound(predef_ff_angle,dim=1)-1 .and. i_i
          endif  !  (N_predef_ff_angles > 0) then
      else if (i_type==3) then
         do iii2 = 1, N_predef_ff_angles
          if (trim(predef_ff_angle(iii2)%name)==trim(the_words(j,5)%ch(1:NNN))) then
          l_ID = .true.
              angle_types(1,i1) = predef_ff_angle(iii2)%style
              prm_angle_types(1:MX_ANGLE_PRMS,i1) = predef_ff_angle(iii2)%the_params(1:MX_ANGLE_PRMS)
          goto 33
          endif ! (trim(predef_ff_angle(iii2)%name)==trim(the_words(j,4)%ch(1:NNN)))
        enddo  ! do iii2 = 1, N_predef_ff_angles

33 continue
        
      else  ! i_type ==2 
         print*, 'ERROR in file', trim(nf), ' a real number was not expected at line', where_in_file(j), &
         ' but expected is an integer or a text defining the angles'
         STOP
      endif

      
      if (.not.l_ID) then ! the angle is defined here and not in ff
            NNN = the_words(j,5)%length
            if (.not.is_angle(trim(the_words(j,5)%ch(1:NNN)))) then
               print*, 'ERROR in file ', trim(nf), ' at line ', where_in_file(j), ' NO ANGLE was defined at 5th coloumn'
               STOP
            endif
            call get_style_angle(trim(the_words(j,5)%ch(1:NNN)),Nprms,angle_types(1,i1))
            call locate_word_in_key(trim(the_words(j,5)%ch(1:NNN)),Nprms,the_words(j,:), l_found,kkk) ! for validation
            do ijk = 1,Nprms
               NNN = the_words(j,kkk+ijk)%length
               call attempt_real_strict(NNN,the_words(j,kkk+ijk)%ch(1:NNN), &
                                        prm_angle_types(ijk,i1)  , trim(nf), where_in_file(j))
            enddo
      endif ! not.l_ID




endif!
enddo ! i


end subroutine read_type_angles

!\see if constrain
             

