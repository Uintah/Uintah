 subroutine get_type_box_in_config
 use file_names_data, only : path_out, FILE_continuing_jobs_indexing,MAX_CH_size
 use chars, only : search_file_for_starting_word, char_intN_ch
 implicit none
 character(4) ch4,ch_4_1,ch_4_2
 integer i_i_1,i_i_2,NN0,i_index,iostat
 logical l_found
 character(MAX_CH_size) nf
  open(unit=167,file=trim(trim(path_out)//trim(FILE_continuing_jobs_indexing)),status='old',iostat=iostat)
   if (iostat == 0 ) then
      read(167,*) i_i_1,i_i_2
      call char_intN_ch(4,i_i_1,ch4)
     if (ch4(1:1).eq.'0') then
      ch4(1:1)=' '
      NN0=2
     endif
     if (ch4(1:1).eq.' '.and.ch4(2:2).eq.'0') then
      ch4(1:1) = ' '; ch4(2:2) =' ' ; NN0=3
     endif
     if (ch4(1:1).eq.' '.and.ch4(2:2).eq.' '.and.ch4(3:3).eq.'0') then
       ch4(1:1) = ' '; ch4(2:2) =' '; ch4(3:3)=' ' ; NN0=4
     endif
     ch_4_1(1:4)=' '; ch_4_1(1:4-NN0+1) = ch4(NN0:4)
     call char_intN_ch(4,i_i_2,ch_4_2)
     nf = trim(path_out)//'config'//'_'//trim(ch_4_1)//'_'//trim(ch_4_2)
   else
     nf = trim(path_out)//'config'
   endif

   open(unit=168,file=trim(nf),status='old',iostat=iostat)
   if (iostat /= 0) then
     write(6,*) 'The attempted config file "',trim(nf),'" does not exist in directory "',trim(path_out),'"',&
     'generate a config file and restart the program'
     STOP
   endif
   close(168)
   call search_file_for_starting_word('SIM_BOX', trim(nf),i_index,l_found)
    if (.not.l_found) then
         print*, 'The attepmted config file "',trim(nf),&
                  '"does not have the keyword INDEX_CONTINUE_JOB '
         print*, 'Make sure you use the correct version of config file ; add INDEX_CONTINUE_JOB to there '&
                 ' and restart the job'
         STOP
   else
        open(unit=168,file=trim(nf))
        do i = 1, i_index;  read(168,*);   enddo
        read(168,*) ; read(168,*); read(168,*) ;
        read(168,*) i_boundary_CTRL
   endif
  close(168)
  close(167)
! \\\\I need to see the symetry of the box (from config)

 end subroutine get_type_box_in_config

