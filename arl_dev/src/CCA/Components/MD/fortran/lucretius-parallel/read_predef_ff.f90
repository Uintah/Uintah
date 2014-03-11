subroutine read_predef_ff_dummies(nf,words,SizeOfLine,NumberOfWords)
use types_module, only : word_type,two_I_one_L_type
use force_field_data
use chars, only : search_words_gi,locate_word_in_key,attempt_real_strict,attempt_integer_strict,&
                  UP_CASE, locate_UPCASE_word_in_key
use max_sizes_data, only :  Max_DummyStyle_Name_Len
use file_names_data, only : MAX_CH_SIZE
implicit none
character(*), intent(IN) :: nf
type(word_type) :: words(:,:)
integer, intent(IN) :: NumberOfWords(:),SizeOfLine(:)
integer Max_words_per_line
integer how_many
integer i,j,k,i1,i2,kkk,i_start,i_end,unit_type,NNN,lines,ijk
logical, allocatable :: l_skip_line(:)
integer, allocatable :: where_in_file(:)
type(two_I_one_L_type), allocatable :: where_starts(:),where_ends(:)
logical l_found,l_found1
character(MAX_ch_size) ch_start,ch_end
i_start = lbound(words,dim=1)
i_end = ubound(words,dim=1)
Max_words_per_line= ubound(words,dim=2) - lbound(words,dim=2) + 1
lines = i_end - i_start + 1
allocate(l_skip_line(i_start:i_end)); l_skip_line=.false.
allocate(where_in_file(i_start:i_end));
allocate(where_starts(i_start:i_end),where_ends(i_start:i_end))

i1=0; i2=0
do i = i_start,i_end
  NNN = words(i,1)%length
  ch_start = ' ';
  do k = 1, NNN
   ch_start(k:k) = UP_CASE(words(i,1)%ch(k:k))
  enddo
  if (ch_start(1:NNN) == 'DEFINE_DUMMY') i1 = i1 + 1
  if (ch_start(1:NNN) == 'END_DEFINE_DUMMY') i2 = i2 + 1
enddo

if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  DEFINE_DUMMY not equal with END_DEFINE_DUMMY',i1,i2
 STOP
endif

how_many = i1
if (how_many==0) then
 print*, 'No DUMMY TYPE was defined in force field'
 N_predef_ff_dummies = 0
 RETURN
endif
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'DEFINE_DUMMY',l_skip_line,how_many,where_starts,.true.)
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'END_DEFINE_DUMMY',l_skip_line,how_many,where_ends,.true.)

 do i = 1, how_many
   if (where_starts(i)%line > where_ends(i)%line) then
      print*, 'ERROR in the file:',trim(nf), ' END_DEFINE_DUMMY cames before DEFINE_DUMMY for dummy type',i
      STOP
   endif
 enddo
 i1 = 0
 do i = 1, how_many
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start = ' ';
     do k = 1, NNN
        ch_start(k:k) = UP_CASE(words(j,1)%ch(k:k))
      enddo
     if (ch_start(1:NNN) == 'DMY') then
         i1 = i1 + 1
     endif
  enddo
 enddo

N_predef_ff_dummies = i1
print*,'N_predef_ff_dummies=',N_predef_ff_dummies
if (allocated(predef_ff_dummy)) deallocate(predef_ff_dummy)
allocate(predef_ff_dummy(N_predef_ff_dummies))
if (allocated(where_in_file)) deallocate(where_in_file)
allocate(where_in_file(N_predef_ff_dummies))
call initialize_dummy_ff(predef_ff_dummy)

i1=0
 do i = 1, how_many
 j=where_starts(i)%line
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start(:) = ' '
     do ijk = 1,NNN
      ch_start(ijk:ijk) = UP_CASE(words(j,1)%ch(ijk:ijk))
     enddo
     if (ch_start(1:NNN) == 'DMY') then
         i1 = i1 + 1
         where_in_file(i1) = j
         if (NumberOfWords(j)<6) then
             print*, 'ERROR in file:',trim(nf), ' at line ',j, ' 6 records needed'
             STOP
         endif
! name
       NNN = words(j,2)%length
       if (NNN > Max_DummyStyle_Name_Len) then
          print*, 'WARNING: The name of the DUMMY defined in file:',trim(nf), 'at line',j,' will be truncated to ',Max_DummyStyle_Name_Len,' characters'
          NNN = Max_DummyStyle_Name_Len
       endif
       predef_ff_dummy(i1)%name(:) = ' '
       predef_ff_dummy(i1)%name(1:NNN) =  words(j,2)%ch(1:NNN)
! style 
       predef_ff_dummy(i1)%style = i1
! LpGeomType
       kkk = 3
       call attempt_integer_strict(words(j,kkk)%length,&
             words(j,kkk)%ch(1:words(j,kkk)%length), &
             predef_ff_dummy(i1)%GeomType, trim(nf), j)
        do ijk = 1,3 
          kkk = 3 + ijk
          call attempt_real_strict(words(j,kkk)%length,&
             words(j,kkk)%ch(1:words(j,kkk)%length), &
             predef_ff_dummy(i1)%the_params(ijk), trim(nf), j)          
        enddo
     endif ! 
  enddo !j
 enddo ! i


do i = 1, N_predef_ff_dummies
     ch_start(:) = ' '
     do  ijk = 1, len(trim(predef_ff_dummy(i)%name))
       ch_start(ijk:ijk) = UP_CASE(trim(predef_ff_dummy(i)%name(ijk:ijk)))
     enddo
     if (trim(ch_start)=='*DMY') then
        print*, 'ERROR in file ', trim(nf), ' at line ', where_in_file(i), ' the second record cannot be *dmy'
        STOP
     endif
do j = i+1,N_predef_ff_dummies
 if (trim(predef_ff_dummy(i)%name) == trim(predef_ff_dummy(j)%name) ) then
    print*, 'ERROR in file ', trim(nf), ' two dumies definitions have the same name see at lines', where_in_file(i),where_in_file(j)
    STOP
 endif
enddo
enddo

deallocate(l_skip_line); 
deallocate(where_in_file);
deallocate(where_starts,where_ends)

end subroutine read_predef_ff_dummies


subroutine read_predef_ff_atoms(nf,words,SizeOfLine,NumberOfWords)
use types_module, only : word_type,two_I_one_L_type
use force_field_data
use chars, only : search_words_gi,locate_word_in_key,attempt_real_strict,attempt_integer_strict,&
                  UP_CASE, locate_UPCASE_word_in_key
use max_sizes_data, only : Max_Atom_Name_Len, Max_vdwStyle_Name_Len
use units_def
use vdw_def
use file_names_data, only : MAX_CH_SIZE
implicit none
character(*), intent(IN) :: nf
type(word_type) :: words(:,:)
integer, intent(IN) :: NumberOfWords(:),SizeOfLine(:)
integer Max_words_per_line
integer how_many
integer i,j,k,i1,i2,kkk,i_start,i_end,unit_type,NNN,lines,ijk
logical, allocatable :: l_skip_line(:)
integer, allocatable :: where_in_file(:)
type(two_I_one_L_type), allocatable :: where_starts(:),where_ends(:)
logical l_found,l_found1
character(MAX_ch_size) ch_start,ch_end
logical l_1

i_start = lbound(words,dim=1)
i_end = ubound(words,dim=1)
Max_words_per_line= ubound(words,dim=2) - lbound(words,dim=2) + 1
lines = i_end - i_start + 1
allocate(l_skip_line(i_start:i_end)); l_skip_line=.false.
allocate(where_in_file(i_start:i_end));
allocate(where_starts(i_start:i_end),where_ends(i_start:i_end))

i1=0; i2=0
do i = i_start,i_end
  NNN = words(i,1)%length
  ch_start = ' '; 
  do k = 1, NNN
   ch_start(k:k) = UP_CASE(words(i,1)%ch(k:k))
  enddo 
  if (ch_start(1:NNN) == 'DEFINE_ATOMS') i1 = i1 + 1
  if (ch_start(1:NNN) == 'END_DEFINE_ATOMS') i2 = i2 + 1
enddo

if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  DEFINE_ATOMS not equal with END_DEFINE_ATOMS',i1,i2
 STOP
endif

how_many = i1
if (how_many==0) then
 print*, 'ERROR:: in the file:',trim(nf), ' No ATOM was defined'
 STOP
endif 


 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'DEFINE_ATOMS',l_skip_line,how_many,where_starts,.true.)
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'END_DEFINE_ATOMS',l_skip_line,how_many,where_ends,.true.)

 do i = 1, how_many
   if (where_starts(i)%line > where_ends(i)%line) then
      print*, 'ERROR in the file:',trim(nf), ' END_DEFINE_ATOMS cames before DEFINE_ATOMS for atom',i
      STOP
   endif
 enddo
 i1 = 0
 do i = 1, how_many
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start = ' ';
     do k = 1, NNN
        ch_start(k:k) = UP_CASE(words(j,1)%ch(k:k))
      enddo
     if (ch_start(1:NNN) == 'ATOM') then
         i1 = i1 + 1
     endif
  enddo
 enddo
 
N_predef_ff_atoms = i1
print*,'N_predef_ff_atoms=',N_predef_ff_atoms
if (allocated(predef_ff_atom)) deallocate(predef_ff_atom)
allocate(predef_ff_atom(N_predef_ff_atoms))
if (allocated(where_in_file)) deallocate(where_in_file) 
allocate(where_in_file(N_predef_ff_atoms))
call initialize_atom_ff(predef_ff_atom)

i1=0
 do i = 1, how_many
 j=where_starts(i)%line
  unit_type = 1 ! default one
  if (NumberOfWords(j) > 1) then ! search for units
    NNN = words(j,2)%length
    call get_units_flag(words(j,2)%ch(1:NNN),unit_type)
  endif
  do j = where_starts(i)%line, where_ends(i)%line
     
     NNN = words(j,1)%length
     ch_start(:) = ' '
     do ijk = 1,NNN
      ch_start(ijk:ijk) = UP_CASE(words(j,1)%ch(ijk:ijk))
     enddo
     if (ch_start(1:NNN) == 'ATOM') then
         i1 = i1 + 1
         where_in_file(i1) = j
         if (NumberOfWords(j)<3) then 
             print*, 'ERROR in file:',trim(nf), ' at line ',j, ' more records needed'
             STOP
         endif
! the units
      predef_ff_atom(i1)%self_vdw%units = unit_type
! name
       NNN = words(j,2)%length  
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the atom defined in file:',trim(nf), 'at line',j,' will be truncated to ',Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_atom(i1)%name(:) = ' '
       predef_ff_atom(i1)%name(1:NNN) =  words(j,2)%ch(1:NNN)
! The mass 
      call attempt_real_strict(words(j,2+1)%length,&
             words(j,2+1)%ch(1:words(j,2+1)%length), &
             predef_ff_atom(i1)%mass , trim(nf), j)
! The charge
       predef_ff_atom(i1)%Q = 0.0d0
       call locate_UPCASE_word_in_key('CH',1,words(j,:), l_found,kkk)
       if (l_found)  then
         call attempt_real_strict(words(j,kkk+1)%length,&
             words(j,kkk+1)%ch(1:words(j,kkk+1)%length), &
             predef_ff_atom(i1)%Q , trim(nf), j)
       endif
! The gauss width
       predef_ff_atom(i1)%QGaussWidth = 1.0d-90
       predef_ff_atom(i1)%isQdistributed = .false.
       call locate_word_in_key('gch',1,words(j,:), l_found,kkk)
       if (l_found) then
          call attempt_real_strict(words(j,kkk+1)%length,&
             words(j,kkk+1)%ch(1:words(j,kkk+1)%length), &
             predef_ff_atom(i1)%QGaussWidth, trim(nf), j)
           predef_ff_atom(i1)%isQdistributed = .true.
       endif
! The charge polarization
       predef_ff_atom(i1)%Qpol = 0.0d0
       predef_ff_atom(i1)%isQpol = .false.
       call locate_UPCASE_word_in_key('*CH',1,words(j,:), l_found,kkk)
       if (l_found) then
          call attempt_real_strict(words(j,kkk+1)%length,&
             words(j,kkk+1)%ch(1:words(j,kkk+1)%length), &
             predef_ff_atom(i1)%Qpol, trim(nf), j)
           predef_ff_atom(i1)%isQpol = (predef_ff_atom(i1)%Qpol > 1.0d-10)
       endif
!  The Dipole
       predef_ff_atom(i1)%Dip = 0.0d0
       call locate_UPCASE_word_in_key('DIP',1,words(j,:), l_found,kkk)
       if (l_found) then
          call attempt_real_strict(words(j,kkk+1)%length,&
             words(j,kkk+1)%ch(1:words(j,kkk+1)%length), &
             predef_ff_atom(i1)%Dip, trim(nf), j)
       endif       
! The initial DipDir
       predef_ff_atom(i1)%DipDir(1) = 0.0d0; predef_ff_atom(i1)%DipDir(2)=0.0d0; predef_ff_atom(i1)%DipDir(3)=1.0d0
       call locate_UPCASE_word_in_key('DIPDIR',3,words(j,:), l_found,kkk)
       if (l_found) then
          call attempt_real_strict(words(j,kkk+1)%length,&
             words(j,kkk+1)%ch(1:words(j,kkk+1)%length), &
             predef_ff_atom(i1)%DipDir(1), trim(nf), j)
          call attempt_real_strict(words(j,kkk+2)%length,&
             words(j,kkk+2)%ch(1:words(j,kkk+2)%length), &
             predef_ff_atom(i1)%DipDir(2), trim(nf), j)
          call attempt_real_strict(words(j,kkk+3)%length,&
             words(j,kkk+3)%ch(1:words(j,kkk+3)%length), &
             predef_ff_atom(i1)%DipDir(3), trim(nf), j)
       endif                   
! The Dipole Polarization
       predef_ff_atom(i1)%DipPol = 0.0d0
       predef_ff_atom(i1)%isDipPol = .false.
       call locate_UPCASE_word_in_key('*DIP',1,words(j,:), l_found,kkk)
       if (l_found) then
          call attempt_real_strict(words(j,kkk+1)%length,&
             words(j,kkk+1)%ch(1:words(j,kkk+1)%length), &
             predef_ff_atom(i1)%DipPol, trim(nf), j)
           predef_ff_atom(i1)%isDipPol = predef_ff_atom(i1)%DipPol > 1.0d-10
       endif            
! is WALL_1?
       predef_ff_atom(i1)%isWaLL_1  = .false.
       call locate_UPCASE_word_in_key('WALL_1',0,words(j,:), l_found,kkk)     
       if (l_found) predef_ff_atom(i1)%isWaLL_1  = .true.     
! do stat on it?
        predef_ff_atom(i1)%more_logic(1) = .false.
       call locate_UPCASE_word_in_key('+STAT',0,words(j,:), l_found,kkk)
       if (l_found) predef_ff_atom(i1)%more_logic(1) = .true.
! self vdw
       predef_ff_atom(i1)%is_self_vdw_Def = .false.
       call locate_UPCASE_word_in_key('VDW',3,words(j,:), l_found,kkk)
       if (l_found) then
          predef_ff_atom(i1)%is_self_vdw_Def = .true.
          NNN = words(j,kkk+1)%length  
          if (NNN > Max_vdwStyle_Name_Len) then
           print*, 'WARNING: The name of the self vdw defined in file:',trim(nf), 'at line',j,' will be truncated to ',Max_vdwStyle_Name_Len,' characters'
          NNN = Max_vdwStyle_Name_Len
          endif
       predef_ff_atom(i1)%self_vdw%StyleName(:) = ' '
       predef_ff_atom(i1)%self_vdw%StyleName(1:NNN) =  words(j,kkk+1)%ch(1:NNN)
       predef_ff_atom(i1)%self_vdw%atom_1_name = trim(predef_ff_atom(i1)%name)
       predef_ff_atom(i1)%self_vdw%atom_2_name = trim(predef_ff_atom(i1)%name)
       call get_style_vdw(trim(predef_ff_atom(i1)%self_vdw%StyleName(1:NNN)),&
                           predef_ff_atom(i1)%self_vdw%N_params,&
                           predef_ff_atom(i1)%self_vdw%style)
       call locate_word_in_key('vdw',1+predef_ff_atom(i1)%self_vdw%N_params,words(j,:), l_found1,kkk) ! for validation
       if (NumberOfWords(j) < kkk+1+predef_ff_atom(i1)%self_vdw%N_params - 1) then
         print*, 'ERROR in file',trim(nf), 'at line ',j,' More records needed to define the vdw interaction'
         STOP
       endif
       do k = kkk+1, kkk+1+predef_ff_atom(i1)%self_vdw%N_params - 1
          call attempt_real_strict(words(j,k+1)%length,&
             words(j,k+1)%ch(1:words(j,k+1)%length), &
             predef_ff_atom(i1)%self_vdw%the_params(k-kkk), trim(nf), j)
       enddo
       endif   

!  DUMMY 
       call locate_UPCASE_word_in_key('*DMY',1,words(j,:), l_found,kkk)
       if (l_found) then
         NNN = words(j,kkk+1)%length
         l_1 = .false.
         do k = 1, N_predef_ff_dummies
             if (trim(predef_ff_dummy(k)%name) == words(j,kkk+1)%ch(1:NNN) ) then
                l_1 = .true.
                predef_ff_atom(i1)%dummy%name = predef_ff_dummy(k)%name
                predef_ff_atom(i1)%dummy%Style = predef_ff_dummy(k)%Style
                predef_ff_atom(i1)%dummy%the_params(:) = predef_ff_dummy(k)%the_params(:)
                predef_ff_atom(i1)%dummy%GeomType      = predef_ff_dummy(k)%GeomType
                predef_ff_atom(i1)%is_dummy = .true.
             endif
         enddo
         if (.not.l_1) then
             print*, 'ERROR in file ',trim(nf), 'at line ',j, ' record ',kkk+1, ' does not correspond to any predefined dummy type '
             STOP
         endif
       endif
       call locate_UPCASE_word_in_key('DMY',1,words(j,:), l_found1,kkk)
       if (l_found.and.l_found1) then
         print*, ' ERROR in file ',trim(nf), 'at line ',j, ' *DMY and DMY cannot be in the same line'
         STOP
       endif
       if (l_found1) then
         call locate_UPCASE_word_in_key('DMY',4,words(j,:), l_found1,kkk)
         kkk = kkk + 1
         call attempt_integer_strict(words(j,kkk)%length,&
             words(j,kkk)%ch(1:words(j,kkk)%length), &
             predef_ff_atom(i1)%dummy%GeomType, trim(nf), j)
          do ijk = 1,3
           kkk = kkk + 1
           call attempt_real_strict(words(j,kkk)%length,&
             words(j,kkk)%ch(1:words(j,kkk)%length), &
             predef_ff_atom(i1)%dummy%the_params(ijk), trim(nf), j)
          enddo
          predef_ff_atom(i1)%is_dummy = .true.
       endif
!  DONE with dummy

     endif !i the validation
  enddo
 enddo

! Validate 2 atoms not to have the same name

do i = 1, N_predef_ff_atoms
do j = i+1,N_predef_ff_atoms
   if ( trim(predef_ff_atom(i)%name) == trim(predef_ff_atom(j)%name) ) then
     print*, 'ERROR in file:',trim(nf), ' when define ATOMS; 2 atoms have the same name; see lines',where_in_file(i),where_in_file(j),&
             ' and either remove one of those lines or change the name of one of the atoms at those 2 lines'
     STOP
   endif
enddo
enddo

 deallocate(where_in_file,l_skip_line)
 deallocate(where_starts,where_ends)

end subroutine read_predef_ff_atoms
! -----------------------------------------
subroutine read_predef_ff_vdw(nf,words,SizeOfLine,NumberOfWords)
use types_module, only : word_type,two_I_one_L_type
use force_field_data
use chars, only : search_words_gi,locate_word_in_key,attempt_real_strict,attempt_integer_strict,&
                  UP_CASE, locate_UPCASE_word_in_key
use max_sizes_data, only : Max_Atom_Name_Len, Max_vdwStyle_Name_Len
use units_def
use vdw_def
use file_names_data, only : MAX_CH_SIZE
implicit none
character(*), intent(IN) :: nf
type(word_type) :: words(:,:)
integer, intent(IN) :: NumberOfWords(:),SizeOfLine(:)
integer Max_words_per_line
integer i,j,k,i1,i2,kkk,i_start,i_end,lines,how_many,NNN,iself,unit_type,ijk
logical, allocatable :: l_skip_line(:)
integer, allocatable :: where_in_file(:)
type(two_I_one_L_type), allocatable :: where_starts(:),where_ends(:)
logical l_found,l_found1
character(MAX_CH_SIZE) ch_start


i_start = lbound(words,dim=1)
i_end = ubound(words,dim=1)
Max_words_per_line= ubound(words,dim=2) - lbound(words,dim=2) + 1
lines = i_end - i_start + 1
allocate(l_skip_line(i_start:i_end)); l_skip_line=.false.
allocate(where_in_file(i_start:i_end));
allocate(where_starts(i_start:i_end),where_ends(i_start:i_end))

i1=0; i2=0
do i = i_start,i_end
  NNN = words(i,1)%length
  do k = 1, NNN
    ch_start(k:k) = UP_CASE(words(i,1)%ch(k:k))
  enddo
  if (ch_start(1:NNN) == 'DEFINE_VDW') i1 = i1 + 1
  if (ch_start(1:NNN) == 'END_DEFINE_VDW') i2 = i2 + 1
enddo

if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  DEFINE_VDW not equal with END_DEFINE_VDW',i1,i2
 STOP
endif

how_many = i1

if (how_many==0) then
 print*, 'ERROR:: in the file:',trim(nf), ' No vdw was defined'
 STOP
endif 


 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'DEFINE_VDW',l_skip_line,how_many,where_starts,.true.)
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'END_DEFINE_VDW',l_skip_line,how_many,where_ends,.true.)

 do i = 1, how_many
   if (where_starts(i)%line > where_ends(i)%line) then
      print*, 'ERROR in the file:',trim(nf), ' END_DEFINE_VDW cames before DEFINE_VDW for atom',i
      STOP
   endif
 enddo

i1 = 0
 do i = 1, how_many
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     do ijk = 1, NNN
       ch_start(ijk:ijk) = UP_CASE(words(j,1)%ch(ijk:ijk))
     enddo
     if (ch_start(1:NNN) == 'VDW') then
         i1 = i1 + 1
     endif
  enddo
 enddo 
 N_predef_ff_vdw = i1 
print*, 'N_predef_ff_vdw=',N_predef_ff_vdw
 iself = 0
 do i=1,N_predef_ff_atoms  ! this is why the atoms needs to be read before vdw
   if (predef_ff_atom(i)%is_self_vdw_Def) then
      iself = iself + 1
   endif
 enddo

print*, 'iself + N_predef_ff_vdw=',iself,N_predef_ff_vdw,iself + N_predef_ff_vdw
allocate(predef_ff_vdw(iself + N_predef_ff_vdw))


 iself = 0
 do i=1,N_predef_ff_atoms  ! this is why the atoms needs to be read before vdw
   if (predef_ff_atom(i)%is_Self_vdw_Def) then
      iself = iself + 1
      call overwrite_2ffvdw(predef_ff_vdw(iself),predef_ff_atom(i)%self_vdw)
   endif
 enddo

i1=iself
 do i = 1, how_many
  j=where_starts(i)%line
  unit_type = 1 ! default one
  if (NumberOfWords(j) > 1) then ! search for units
    NNN = words(j,2)%length
    call get_units_flag(words(j,2)%ch(1:NNN),unit_type)
  endif
  do j = where_starts(i)%line, where_ends(i)%line   
     NNN = words(j,1)%length
     if (words(j,1)%ch(1:NNN) == 'vdw') then
         i1 = i1 + 1
         where_in_file(i1) = j
         if (NumberOfWords(j)<4+2) then 
             print*, 'ERROR in file:',trim(nf), ' at line ',j, ' more records needed'
             STOP
         endif
! the units
      predef_ff_vdw(i1)%units = unit_type
! NameStyle
        NNN = words(j,2)%length  
       if (NNN > Max_vdwStyle_Name_Len) then
          print*, 'WARNING: The name of the vdwStyle in file:',trim(nf), 'at line',j,' will be truncated to ',Max_vdwStyle_Name_Len,' characters'
          NNN = Max_vdwStyle_Name_Len
       endif
       predef_ff_vdw(i1)%StyleName(:) = ' '
       predef_ff_vdw(i1)%StyleName(1:NNN) =  words(j,2)%ch(1:NNN)
       call get_style_vdw(trim(predef_ff_vdw(i1)%StyleName(1:NNN)),&
                           predef_ff_vdw(i1)%N_params,&
                           predef_ff_vdw(i1)%style)
! Name Atom 1
        NNN = words(j,3)%length  
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the vdwStyle in file:',trim(nf), 'at line',j,' will be truncated to ',Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_vdw(i1)%atom_1_name(:) = ' '
       predef_ff_vdw(i1)%atom_1_name(1:NNN) =  words(j,3)%ch(1:NNN)
! Name Atom 2
        NNN = words(j,4)%length  
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the vdwStyle in file:',trim(nf), 'at line',j,' will be truncated to ',Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_vdw(i1)%atom_2_name(:) = ' '
       predef_ff_vdw(i1)%atom_2_name(1:NNN) =  words(j,4)%ch(1:NNN)
       call locate_UPCASE_word_in_key('VDW',1+2+predef_ff_vdw(i1)%N_params,words(j,:), l_found1,kkk) ! for validation
! get the parameters

       
       do k = kkk+3, kkk+3+predef_ff_vdw(i1)%N_params-1
          call attempt_real_strict(words(j,k+1)%length,&
             words(j,k+1)%ch(1:words(j,k+1)%length), &
             predef_ff_vdw(i1)%the_params(k-kkk-2), trim(nf), j)
       enddo
 endif
 enddo
 enddo 
    

! Put them in default (kJ/mol) input units
 do i =  1, N_predef_ff_vdw+iself
   call get_vdw_units(&
   predef_ff_vdw(i)%units,&
   predef_ff_vdw(i)%style,&
   predef_ff_vdw(i)%N_params,&
   predef_ff_vdw(i)%the_params)
 enddo
 do i = 1, N_predef_ff_atoms
  if (predef_ff_atom(i)%is_self_vdw_Def) then
   call get_vdw_units(&
   predef_ff_atom(i)%self_vdw%units,&
   predef_ff_atom(i)%self_vdw%style,&
   predef_ff_atom(i)%self_vdw%N_params,&
   predef_ff_atom(i)%self_vdw%the_params)
  endif   ! Make them in kJ/mol
 enddo



 
  ! Validate 2 vdw not to defined twice: same style; same atom1; same atom2
 N_predef_ff_vdw = N_predef_ff_vdw + iself 
do i = iself+1, N_predef_ff_vdw
do j = i+1,N_predef_ff_vdw
   if ( predef_ff_vdw(i)%style == predef_ff_vdw(j)%style ) then
   if ( trim(predef_ff_vdw(i)%atom_1_name) == trim(predef_ff_vdw(j)%atom_1_name) ) then 
   if ( trim(predef_ff_vdw(i)%atom_2_name) == trim(predef_ff_vdw(j)%atom_2_name) ) then 
     print*, 'ERROR in file:',trim(nf), ' when define vdw; 2 vdw are identical; see lines',where_in_file(i),where_in_file(j),&
             ' and either remove one of those lines or see if there is a mistake defining vdw; see the file lines:',&
             where_in_file(i),where_in_file(j)
     STOP
   endif
   endif
   endif
   
   if ( predef_ff_vdw(i)%style == predef_ff_vdw(j)%style ) then
   if ( trim(predef_ff_vdw(i)%atom_1_name) == trim(predef_ff_vdw(j)%atom_2_name) ) then 
   if ( trim(predef_ff_vdw(i)%atom_2_name) == trim(predef_ff_vdw(j)%atom_1_name) ) then 
     print*, 'ERROR in file:',trim(nf), ' when define vdw; 2 vdw are identical; see lines',where_in_file(i),where_in_file(j),&
             ' and either remove one of those lines or see if there is a mistake defining vdw; see the file lines:',&
             where_in_file(i),where_in_file(j)
     STOP
   endif
   endif
   endif
   
enddo   
enddo    

 ! Compare self interactions with vdw
 
 do i = 1,iself
 do j = iself+1,N_predef_ff_vdw
   if ( predef_ff_vdw(i)%style == predef_ff_vdw(j)%style ) then
   if ( trim(predef_ff_vdw(i)%atom_1_name) == trim(predef_ff_vdw(j)%atom_1_name) ) then 
   if ( trim(predef_ff_vdw(i)%atom_2_name) == trim(predef_ff_vdw(j)%atom_2_name) ) then 
     print*, 'WARNING in file:',trim(nf), ' when define vdw;  2 self-vdw are defined twice: at atom definition and at vdw definition',&
            'The atom definition will be overwritten by the last one; see the correspondign vdw that is kept in line:', where_in_file(j)
     call overwrite_2ffvdw(predef_ff_vdw(i),predef_ff_vdw(j))
   endif
   endif
   endif
   ! DO IT ONLY ONCE BECAUSE THEY ARE SELF VDW
 enddo
 enddo

  
 deallocate(where_in_file,l_skip_line)    
 deallocate(where_starts,where_ends)

 
end subroutine read_predef_ff_vdw

subroutine read_predef_ff_bonds(nf,words,SizeOfLine,NumberOfWords)
use types_module, only : word_type,two_I_one_L_type
use force_field_data
use chars, only : search_words_gi,locate_word_in_key,attempt_real_strict,attempt_integer_strict,UP_CASE, &
                  locate_UPCASE_word_in_key
use max_sizes_data, only : Max_Atom_Name_Len, Max_vdwStyle_Name_Len
use file_names_data, only :  MAX_CH_size
use units_def
use intramol_forces_def

implicit none
character(*), intent(IN) :: nf
type(word_type) :: words(:,:)
integer, intent(IN) :: NumberOfWords(:),SizeOfLine(:)
integer Max_words_per_line
integer how_many
integer i,j,k,i1,i2,kkk,i_start,i_end,unit_type,NNN,lines,ijk
logical, allocatable :: l_skip_line(:)
integer, allocatable :: where_in_file(:)
type(two_I_one_L_type), allocatable :: where_starts(:),where_ends(:)
logical l_found,l_found1
character(MAX_ch_size) ch_start,ch_end

i_start = lbound(words,dim=1)
i_end = ubound(words,dim=1)
Max_words_per_line= ubound(words,dim=2) - lbound(words,dim=2) + 1
lines = i_end - i_start + 1
allocate(l_skip_line(i_start:i_end)); l_skip_line=.false.
allocate(where_in_file(i_start:i_end));
allocate(where_starts(i_start:i_end),where_ends(i_start:i_end))

i1=0; i2=0
do i = i_start,i_end
  NNN = words(i,1)%length
  ch_start = ' ';
  do k = 1, NNN
   ch_start(k:k) = UP_CASE(words(i,1)%ch(k:k))
  enddo
  if (ch_start(1:NNN) == 'DEFINE_BONDS') i1 = i1 + 1
  if (ch_start(1:NNN) == 'END_DEFINE_BONDS') i2 = i2 + 1
enddo

if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  DEFINE_BONDS not equal with END_DEFINE_BONDS',i1,i2
 STOP
endif

how_many = i1
if (how_many==0) then
 print*, 'WARNING :: in the file:',trim(nf), ' No BOND (or constrain) was pre-defined'
 RETURN
endif

 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'DEFINE_BONDS',l_skip_line,how_many,where_starts,.true.)
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'END_DEFINE_BONDS',l_skip_line,how_many,where_ends,.true.)

 do i = 1, how_many
   if (where_starts(i)%line > where_ends(i)%line) then
      print*, 'ERROR in the file:',trim(nf), ' END_DEFINE_BONDS cames before DEFINE_BONDS for set ',i
      STOP
   endif
 enddo
 i1 = 0
 do i = 1, how_many
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start = ' ';
     do k = 1, NNN
        ch_start(k:k) = UP_CASE(words(j,1)%ch(k:k))
      enddo
     if (ch_start(1:NNN) == 'BOND') then
         i1 = i1 + 1
     endif
  enddo
 enddo

N_predef_ff_bonds = i1
if (N_predef_ff_bonds==0) RETURN

if (allocated(predef_ff_bond)) deallocate(predef_ff_bond)
allocate(predef_ff_bond(N_predef_ff_bonds))
if (allocated(where_in_file)) deallocate(where_in_file)
allocate(where_in_file(N_predef_ff_bonds))

call initialize_bond_ff(predef_ff_bond)

i1=0
 do i = 1, how_many
 j=where_starts(i)%line
  unit_type = 1 ! default one
  if (NumberOfWords(j) > 1) then ! search for units
    NNN = words(j,2)%length
    call get_units_flag(words(j,2)%ch(1:NNN),unit_type)
  endif
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start(:) = ' '
     do ijk = 1,NNN
      ch_start(ijk:ijk) = UP_CASE(words(j,1)%ch(ijk:ijk))
     enddo
     if (ch_start(1:NNN) == 'BOND') then
         i1 = i1 + 1
         where_in_file(i1) = j
         if (NumberOfWords(j)<6) then
             print*, 'ERROR in file:',trim(nf), ' at line ',j, ' more records needed'
             STOP
         endif
! the units
      predef_ff_bond(i1)%units = unit_type
! name
       NNN = words(j,2)%length
       if (NNN > Max_Bond_Name_Len) then
          print*, 'WARNING: The name of the BOND defined in file:',trim(nf), 'at line',j,' will be truncated to ',Max_BOND_Name_Len,' characters'
          NNN = Max_Bond_Name_Len
       endif
       predef_ff_bond(i1)%name(1:NNN) = words(j,2)%ch(1:NNN)

       NNN = words(j,3)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within BOND defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_bond(i1)%atom_1_name(1:NNN) = words(j,3)%ch(1:NNN)
       NNN = words(j,4)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within BOND defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_bond(i1)%atom_2_name(1:NNN) = words(j,4)%ch(1:NNN)

       NNN = words(j,5)%length
       do ijk=1,NNN
         ch_start(ijk:ijk) = UP_CASE(words(j,5)%ch(ijk:ijk))
       enddo 
       if (NNN > Max_BondStyle_Name_Len) then
          print*, 'WARNING: The name of the StyleBond  within BOND defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_BondStyle_Name_Len,' characters'
          NNN = Max_BondStyle_Name_Len
       endif
       predef_ff_bond(i1)%StyleName(1:NNN) = ch_start(1:NNN)

       call get_style_bond(trim(predef_ff_bond(i1)%StyleName(1:NNN)),&
                                predef_ff_bond(i1)%N_params,predef_ff_bond(i1)%style)  

       call locate_word_in_key(trim(predef_ff_bond(i1)%StyleName(1:NNN)),&
                                predef_ff_bond(i1)%N_params,words(j,:), l_found1,kkk) ! for validation
       
       do ijk = kkk+1, kkk+1+predef_ff_bond(i1)%N_params - 1
           call attempt_real_strict(words(j,ijk)%length,&
             words(j,ijk)%ch(1:words(j,ijk)%length), &
             predef_ff_bond(i1)%the_params(ijk-kkk), trim(nf), j)
       enddo 

       call locate_UPCASE_word_in_key('*CONS',1,words(j,:), l_found1,kkk) ! for validation
       if (l_found1) then
       if (kkk == 2) then
        print*, 'ERROR in file ',trim(nf), ' at line ', j, &
        ' , record 2. The name of the bond must be anything else but not *CONS which define a constrain'
        STOP
       endif 
          ijk = kkk + 1
           call attempt_real_strict(words(j,ijk)%length,&
             words(j,ijk)%ch(1:words(j,ijk)%length), &
             predef_ff_bond(i1)%constrained_bond, trim(nf), j)
         predef_ff_bond(i1)%is_constrain = .true.
       else
         predef_ff_bond(i1)%is_constrain = .false.
         predef_ff_bond(i1)%constrained_bond = predef_ff_bond(i1)%the_params(predef_ff_bond(i1)%N_params)  ! The last parameter is the eq. dist
       endif 

       
       call set_units_bond(trim(predef_ff_bond(i1)%StyleName),&
                            predef_ff_bond(i1)%units, predef_ff_bond(i1)%N_params,&
                            predef_ff_bond(i1)%the_params( 1:predef_ff_bond(i1)%N_params), &
                            predef_ff_bond(i1)%constrained_bond )

!print*, 'i1 bonds = ',i1, ' :',trim(predef_ff_bond(i1)%StyleName), ' ' , trim( predef_ff_bond(i1)%atom_1_name(1:NNN)), ' ' , &
!         trim(predef_ff_bond(i1)%atom_2_name(1:NNN)), ' ', trim(predef_ff_bond(i1)%StyleName(1:NNN)), &
!         'Nparams style=',predef_ff_bond(i1)%N_params,predef_ff_bond(i1)%style, &
!         'the params=',predef_ff_bond(i1)%the_params(1:predef_ff_bond(i1)%N_params),&
!         ' is_constrain constrained_bond=',predef_ff_bond(i1)%is_constrain, predef_ff_bond(i1)%constrained_bond



     endif ! (ch_start(1:NNN) == 'BOND') then
  enddo  ! j
 enddo !i = 1,how_many


do i = 1, N_predef_ff_bonds
 if (ch_start(1:NNN)=='*CONS') then
  print*, ' ERROR when defining the force field bonds; A bond name can be anything exept *CONS which define a constrain'
  print*, ' RENAME the bond defined by *CONS to any other name ; see the file:', trim(nf), 'at line', where_in_file(i)
  STOP 
 endif
do j = i+1,N_predef_ff_bonds
   if ( trim(predef_ff_bond(i)%name) == trim(predef_ff_bond(j)%name) ) then
     print*, 'ERROR in file:',trim(nf), ' when define BONDS; 2 bonds have the same name; see lines',where_in_file(i),where_in_file(j),&
             ' and either remove one of those lines or change the name of one of the bonds at those 2 lines'
     STOP
   endif
enddo
enddo

end subroutine read_predef_ff_bonds

subroutine read_predef_ff_angles(nf,words,SizeOfLine,NumberOfWords)
use types_module, only : word_type,two_I_one_L_type
use force_field_data
use chars, only : search_words_gi,locate_word_in_key,attempt_real_strict,attempt_integer_strict,UP_CASE, &
                  locate_UPCASE_word_in_key
use max_sizes_data, only : Max_Atom_Name_Len, Max_vdwStyle_Name_Len
use file_names_data, only :  MAX_CH_size
use units_def
use intramol_forces_def

implicit none
character(*), intent(IN) :: nf
type(word_type) :: words(:,:)
integer, intent(IN) :: NumberOfWords(:),SizeOfLine(:)
integer Max_words_per_line
integer how_many
integer i,j,k,i1,i2,kkk,i_start,i_end,unit_type,NNN,lines,ijk
logical, allocatable :: l_skip_line(:)
integer, allocatable :: where_in_file(:)
type(two_I_one_L_type), allocatable :: where_starts(:),where_ends(:)
logical l_found,l_found1
character(MAX_ch_size) ch_start,ch_end
i_start = lbound(words,dim=1)
i_end = ubound(words,dim=1)
Max_words_per_line= ubound(words,dim=2) - lbound(words,dim=2) + 1
lines = i_end - i_start + 1
allocate(l_skip_line(i_start:i_end)); l_skip_line=.false.
allocate(where_in_file(i_start:i_end));
allocate(where_starts(i_start:i_end),where_ends(i_start:i_end))

i1=0; i2=0
do i = i_start,i_end
  NNN = words(i,1)%length
  ch_start = ' ';
  do k = 1, NNN
   ch_start(k:k) = UP_CASE(words(i,1)%ch(k:k))
  enddo
  if (ch_start(1:NNN) == 'DEFINE_ANGLES') i1 = i1 + 1
  if (ch_start(1:NNN) == 'END_DEFINE_ANGLES') i2 = i2 + 1
enddo

if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  DEFINE_ANGLES not equal with END_DEFINE_ANGLES',i1,i2
 STOP
endif

how_many = i1
if (how_many==0) then
 print*, 'WARNING :: in the file:',trim(nf), ' No angle (or constrain) was pre-defined'
 RETURN
endif
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'DEFINE_ANGLES',l_skip_line,how_many,where_starts,.true.)
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'END_DEFINE_ANGLES',l_skip_line,how_many,where_ends,.true.)

 do i = 1, how_many
   if (where_starts(i)%line > where_ends(i)%line) then
      print*, 'ERROR in the file:',trim(nf), ' END_DEFINE_angleS cames before DEFINE_angleS for set ',i
      STOP
   endif
 enddo
 i1 = 0
 do i = 1, how_many
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start = ' ';
     do k = 1, NNN
        ch_start(k:k) = UP_CASE(words(j,1)%ch(k:k))
      enddo
     if (ch_start(1:NNN) == 'ANGLE') then
         i1 = i1 + 1
     endif
  enddo
 enddo

N_predef_ff_angles = i1
if (N_predef_ff_angles==0) RETURN
print*,'N_predef_ff_angles=',N_predef_ff_angles

if (allocated(predef_ff_angle)) deallocate(predef_ff_angle)
allocate(predef_ff_angle(N_predef_ff_angles))
if (allocated(where_in_file)) deallocate(where_in_file)
allocate(where_in_file(N_predef_ff_angles))

call initialize_angle_ff(predef_ff_angle)

i1=0
 do i = 1, how_many
 j=where_starts(i)%line
  unit_type = 1 ! default one
  if (NumberOfWords(j) > 1) then ! search for units
    NNN = words(j,2)%length
    call get_units_flag(words(j,2)%ch(1:NNN),unit_type)
  endif
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start(:) = ' '
     do ijk = 1,NNN
      ch_start(ijk:ijk) = UP_CASE(words(j,1)%ch(ijk:ijk))
     enddo
     if (ch_start(1:NNN) == 'ANGLE') then
         i1 = i1 + 1
         where_in_file(i1) = j
         if (NumberOfWords(j)<7) then
             print*, 'ERROR in file:',trim(nf), ' at line ',j, ' more records needed'
             STOP
         endif
! the units
      predef_ff_angle(i1)%units = unit_type
! name
       NNN = words(j,2)%length
       if (NNN > Max_angle_Name_Len) then
          print*, 'WARNING: The name of the angle defined in file:',trim(nf), 'at line',j,' will be truncated to ',Max_angle_Name_Len,' characters'
          NNN = Max_angle_Name_Len
       endif
       predef_ff_angle(i1)%name(1:NNN) = words(j,2)%ch(1:NNN)

       NNN = words(j,3)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within angle defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_angle(i1)%atom_1_name(1:NNN) = words(j,3)%ch(1:NNN)
       NNN = words(j,4)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within angle defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_angle(i1)%atom_2_name(1:NNN) = words(j,4)%ch(1:NNN)

       NNN = words(j,5)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within angle defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_angle(i1)%atom_3_name(1:NNN) = words(j,5)%ch(1:NNN)


       NNN = words(j,6)%length
       do ijk=1,NNN
         ch_start(ijk:ijk) = UP_CASE(words(j,6)%ch(ijk:ijk))
       enddo
       if (NNN > Max_angleStyle_Name_Len) then
          print*, 'WARNING: The name of the Styleangle  within angle defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_angleStyle_Name_Len,' characters'
          NNN = Max_angleStyle_Name_Len
       endif
       predef_ff_angle(i1)%StyleName(1:NNN) = ch_start(1:NNN)

       call get_style_angle(trim(predef_ff_angle(i1)%StyleName(1:NNN)),&
                                predef_ff_angle(i1)%N_params,predef_ff_angle(i1)%style)

       call locate_word_in_key(trim(predef_ff_angle(i1)%StyleName(1:NNN)),&
                                predef_ff_angle(i1)%N_params,words(j,:), l_found1,kkk) ! for validation

       do ijk = kkk+1, kkk+1+predef_ff_angle(i1)%N_params - 1
           call attempt_real_strict(words(j,ijk)%length,&
             words(j,ijk)%ch(1:words(j,ijk)%length), &
             predef_ff_angle(i1)%the_params(ijk-kkk), trim(nf), j)
       enddo

       call set_units_angle(trim(predef_ff_angle(i1)%StyleName),&
                            predef_ff_angle(i1)%units, predef_ff_angle(i1)%N_params,&
                            predef_ff_angle(i1)%the_params( 1:predef_ff_angle(i1)%N_params) )

!print*, 'i1 angles = ',i1, ' :',trim(predef_ff_angle(i1)%StyleName), ' ' , trim( predef_ff_angle(i1)%atom_1_name(1:NNN)), ' ' , &
!         trim(predef_ff_angle(i1)%atom_2_name(1:NNN)), ' ', &
!         trim(predef_ff_angle(i1)%atom_3_name(1:NNN)), ' ',&
!         trim(predef_ff_angle(i1)%StyleName(1:NNN)), &
!         'Nparams style=',predef_ff_angle(i1)%N_params,predef_ff_angle(i1)%style, &
!         'the params=',predef_ff_angle(i1)%the_params(1:predef_ff_angle(i1)%N_params)



     endif ! (ch_start(1:NNN) == 'angle') then
  enddo  ! j
 enddo !i = 1,how_many



do i = 1, N_predef_ff_angles
do j = i+1,N_predef_ff_angles
   if ( trim(predef_ff_angle(i)%name) == trim(predef_ff_angle(j)%name) ) then
     print*, 'ERROR in file:',trim(nf), ' when define ANGLES; 2 angle have the same name; see lines',where_in_file(i),where_in_file(j),&
             ' and either remove one of those lines or change the name of one of the angles at those 2 lines'
     STOP
   endif
enddo
enddo

!print*, 'STOP in read_predef_ff_angles'
!STOP
end subroutine read_predef_ff_angles

subroutine read_predef_ff_dihedrals(nf,words,SizeOfLine,NumberOfWords)
use types_module, only : word_type,two_I_one_L_type
use force_field_data
use chars, only : search_words_gi,locate_word_in_key,attempt_real_strict,attempt_integer_strict,UP_CASE, &
                  locate_UPCASE_word_in_key
use max_sizes_data, only : Max_Atom_Name_Len, Max_vdwStyle_Name_Len
use file_names_data, only :  MAX_CH_size
use units_def
use intramol_forces_def

implicit none
character(*), intent(IN) :: nf
type(word_type) :: words(:,:)
integer, intent(IN) :: NumberOfWords(:),SizeOfLine(:)
integer Max_words_per_line
integer how_many
integer i,j,k,i1,i2,kkk,i_start,i_end,unit_type,NNN,lines,ijk
logical, allocatable :: l_skip_line(:)
integer, allocatable :: where_in_file(:)
type(two_I_one_L_type), allocatable :: where_starts(:),where_ends(:)
logical l_found,l_found1
character(MAX_ch_size) ch_start,ch_end
i_start = lbound(words,dim=1)
i_end = ubound(words,dim=1)
Max_words_per_line= ubound(words,dim=2) - lbound(words,dim=2) + 1
lines = i_end - i_start + 1
allocate(l_skip_line(i_start:i_end)); l_skip_line=.false.
allocate(where_in_file(i_start:i_end));
allocate(where_starts(i_start:i_end),where_ends(i_start:i_end))


i1=0; i2=0
do i = i_start,i_end
  NNN = words(i,1)%length
  ch_start = ' ';
  do k = 1, NNN
   ch_start(k:k) = UP_CASE(words(i,1)%ch(k:k))
  enddo
  if (ch_start(1:NNN) == 'DEFINE_DIHEDRALS') i1 = i1 + 1
  if (ch_start(1:NNN) == 'END_DEFINE_DIHEDRALS') i2 = i2 + 1
enddo

if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  DEFINE_DIHEDRALS not equal with END_DEFINE_DIHEDRALS',i1,i2
 STOP
endif

how_many = i1
if (how_many==0) then
 print*, 'WARNING :: in the file:',trim(nf), ' No DIHEDRAL was pre-defined'
 RETURN
endif
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'DEFINE_DIHEDRALS',l_skip_line,how_many,where_starts,.true.)
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'END_DEFINE_DIHEDRALS',l_skip_line,how_many,where_ends,.true.)

 do i = 1, how_many
   if (where_starts(i)%line > where_ends(i)%line) then
      print*, 'ERROR in the file:',trim(nf), ' END_DEFINE_DIHEDRALS cames before DEFINE_DIHEDRALS for set ',i
      STOP
   endif
 enddo
 i1 = 0
 do i = 1, how_many
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start = ' ';
     do k = 1, NNN
        ch_start(k:k) = UP_CASE(words(j,1)%ch(k:k))
      enddo
     if (ch_start(1:NNN) == 'DIH') then
         i1 = i1 + 1
     endif
  enddo
 enddo

N_predef_ff_dihs = i1
if (N_predef_ff_dihs==0) RETURN

print*,'N_predef_ff_dihss=',N_predef_ff_dihs

if (allocated(predef_ff_dih)) deallocate(predef_ff_dih)
allocate(predef_ff_dih(N_predef_ff_dihs))
if (allocated(where_in_file)) deallocate(where_in_file)
allocate(where_in_file(N_predef_ff_dihs))

call initialize_dih_ff(predef_ff_dih)

i1=0
 do i = 1, how_many
 j=where_starts(i)%line
  unit_type = 1 ! default one
  if (NumberOfWords(j) > 1) then ! search for units
    NNN = words(j,2)%length
    call get_units_flag(words(j,2)%ch(1:NNN),unit_type)
  endif
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start(:) = ' '
     do ijk = 1,NNN
      ch_start(ijk:ijk) = UP_CASE(words(j,1)%ch(ijk:ijk))
     enddo
     if (ch_start(1:NNN) == 'DIH') then
         i1 = i1 + 1
         where_in_file(i1) = j
         if (NumberOfWords(j)<8) then
             print*, 'ERROR in file:',trim(nf), ' at line ',j, ' more records needed'
             STOP
         endif
! the units
      predef_ff_DIH(i1)%units = unit_type
! name
       NNN = words(j,2)%length
       if (NNN > Max_DIH_Name_Len) then
          print*, 'WARNING: The name of the DIH defined in file:',trim(nf), 'at line',j,' will be truncated to ',Max_DIH_Name_Len,' characters'
          NNN = Max_DIH_Name_Len
       endif
       predef_ff_DIH(i1)%name(1:NNN) = words(j,2)%ch(1:NNN)

       NNN = words(j,3)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within DIH defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_DIH(i1)%atom_1_name(1:NNN) = words(j,3)%ch(1:NNN)
       NNN = words(j,4)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within DIH defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_DIH(i1)%atom_2_name(1:NNN) = words(j,4)%ch(1:NNN)
       NNN = words(j,5)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within DIH defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_DIH(i1)%atom_3_name(1:NNN) = words(j,5)%ch(1:NNN)
       NNN = words(j,6)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within DIH defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_DIH(i1)%atom_4_name(1:NNN) = words(j,6)%ch(1:NNN)

       NNN = words(j,7)%length
       do ijk=1,NNN
         ch_start(ijk:ijk) = UP_CASE(words(j,7)%ch(ijk:ijk))
       enddo
       if (NNN > Max_DIHStyle_Name_Len) then
          print*, 'WARNING: The name of the StyleDIH  within DIH defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_DIHStyle_Name_Len,' characters'
          NNN = Max_DIHStyle_Name_Len
       endif
       predef_ff_DIH(i1)%StyleName(1:NNN) = ch_start(1:NNN)

       call locate_word_in_key(trim(predef_ff_DIH(i1)%StyleName(1:NNN)),&
                                predef_ff_DIH(i1)%N_params,words(j,:), l_found1,kkk) ! for validation
       if (trim(predef_ff_DIH(i1)%StyleName(1:NNN)) == 'COS_N') then
          call attempt_integer_strict(words(j,kkk+1)%length,&
             words(j,kkk+1)%ch(1:words(j,kkk+1)%length), &
             predef_ff_DIH(i1)%N_params, trim(nf), j)
          call locate_word_in_key(trim(predef_ff_DIH(i1)%StyleName(1:NNN)),&
                                predef_ff_DIH(i1)%N_params+1,words(j,:), l_found1,kkk)
          kkk = kkk + 1
       endif
        call get_style_dihedral(trim(predef_ff_dih(i1)%StyleName(1:NNN)),&
                                predef_ff_dih(i1)%N_params,predef_ff_dih(i1)%style)


       do ijk = kkk+1, kkk+1+predef_ff_DIH(i1)%N_params - 1
           call attempt_real_strict(words(j,ijk)%length,&
             words(j,ijk)%ch(1:words(j,ijk)%length), &
             predef_ff_DIH(i1)%the_params(ijk-kkk), trim(nf), j)
       enddo

       call set_units_dihedral(trim(predef_ff_dih(i1)%StyleName),&
                            predef_ff_dih(i1)%units, predef_ff_dih(i1)%N_params,&
                            predef_ff_dih(i1)%the_params(:) )

!print*, 'i1 DIHs = ',i1, ' :',trim(predef_ff_DIH(i1)%StyleName), ' ' , trim( predef_ff_DIH(i1)%atom_1_name(1:NNN)), ' ' , &
!         trim(predef_ff_DIH(i1)%atom_2_name(1:NNN)), ' ', &
!         trim(predef_ff_DIH(i1)%atom_3_name(1:NNN)), ' ',&
!         trim(predef_ff_DIH(i1)%atom_4_name(1:NNN)), ' ',&
!         trim(predef_ff_DIH(i1)%StyleName(1:NNN)), &
!         'Nparams style=',predef_ff_DIH(i1)%N_params,predef_ff_DIH(i1)%style, &
!         'the params=',predef_ff_DIH(i1)%the_params(1:predef_ff_DIH(i1)%N_params),&
!         'units = ',predef_ff_DIH(i1)%units
!read(*,*)

     endif ! (ch_start(1:NNN) == 'DIH') then
  enddo  ! j
 enddo !i = 1,how_many


do i = 1, N_predef_ff_dihs
do j = i+1,N_predef_ff_dihs
   if ( trim(predef_ff_dih(i)%name) == trim(predef_ff_dih(j)%name) ) then
     print*, 'ERROR in file:',trim(nf), ' when define DIHEDARLS; 2 dihedrals have the same name; see lines',where_in_file(i),where_in_file(j),&
             ' and either remove one of those lines or change the name of one of the dihedrals at those 2 lines'
     STOP
   endif
enddo
enddo


end subroutine read_predef_ff_dihedrals



subroutine read_predef_ff_deforms(nf,words,SizeOfLine,NumberOfWords)
use types_module, only : word_type,two_I_one_L_type
use force_field_data
use chars, only : search_words_gi,locate_word_in_key,attempt_real_strict,attempt_integer_strict,UP_CASE, &
                  locate_UPCASE_word_in_key
use max_sizes_data, only : Max_Atom_Name_Len
use file_names_data, only :  MAX_CH_size
use units_def
use intramol_forces_def

implicit none
character(*), intent(IN) :: nf
type(word_type) :: words(:,:)
integer, intent(IN) :: NumberOfWords(:),SizeOfLine(:)
integer Max_words_per_line
integer how_many
integer i,j,k,i1,i2,kkk,i_start,i_end,unit_type,NNN,lines,ijk
logical, allocatable :: l_skip_line(:)
integer, allocatable :: where_in_file(:)
type(two_I_one_L_type), allocatable :: where_starts(:),where_ends(:)
logical l_found,l_found1
character(MAX_ch_size) ch_start,ch_end
i_start = lbound(words,dim=1)
i_end = ubound(words,dim=1)
Max_words_per_line= ubound(words,dim=2) - lbound(words,dim=2) + 1
lines = i_end - i_start + 1
allocate(l_skip_line(i_start:i_end)); l_skip_line=.false.
allocate(where_in_file(i_start:i_end));
allocate(where_starts(i_start:i_end),where_ends(i_start:i_end))

i1=0; i2=0
do i = i_start,i_end
  NNN = words(i,1)%length
  ch_start = ' ';
  do k = 1, NNN
   ch_start(k:k) = UP_CASE(words(i,1)%ch(k:k))
  enddo
  if (ch_start(1:NNN) == 'DEFINE_OUTOFPLANEDEFORMS') i1 = i1 + 1
  if (ch_start(1:NNN) == 'END_DEFINE_OUTOFPLANEDEFORMS') i2 = i2 + 1
enddo

if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  DEFINE_DEFORMS not equal with END_DEFINE_DEFORMS',i1,i2
 STOP
endif

how_many = i1
if (how_many==0) then
 print*, 'WARNING :: in the file:',trim(nf), ' No DEFORM was pre-defined'
 RETURN
endif
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'DEFINE_OUTOFPLANEDEFORMS',l_skip_line,how_many,where_starts,.true.)
 call search_words_gi(i_start,i_end,lines,trim(nf),Max_words_per_line,words,SizeOfLine,NumberOfWords,&
                   'END_DEFINE_OUTOFPLANEDEFORMS',l_skip_line,how_many,where_ends,.true.)

 do i = 1, how_many
   if (where_starts(i)%line > where_ends(i)%line) then
      print*, 'ERROR in the file:',trim(nf), ' END_DEFINE_OUTOFPLANEDEFORMS cames before DEFINE_OUTOFPLANEDEFORMS for set ',i
      STOP
   endif
 enddo
 i1 = 0
 do i = 1, how_many
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start = ' ';
     do k = 1, NNN
        ch_start(k:k) = UP_CASE(words(j,1)%ch(k:k))
      enddo
     if (ch_start(1:NNN) == 'DEFORM') then
         i1 = i1 + 1
     endif
  enddo
 enddo

N_predef_ff_deforms = i1
if (N_predef_ff_deforms==0) RETURN

print*,'N_predef_ff_deforms=',N_predef_ff_deforms

if (allocated(predef_ff_deform)) deallocate(predef_ff_deform)
allocate(predef_ff_deform(N_predef_ff_deforms))
if (allocated(where_in_file)) deallocate(where_in_file)
allocate(where_in_file(N_predef_ff_deforms))

call initialize_deform_ff(predef_ff_deform)

i1=0
 do i = 1, how_many
 j=where_starts(i)%line
  unit_type = 1 ! default one
  if (NumberOfWords(j) > 1) then ! search for units
    NNN = words(j,2)%length
    call get_units_flag(words(j,2)%ch(1:NNN),unit_type)
  endif
  do j = where_starts(i)%line, where_ends(i)%line
     NNN = words(j,1)%length
     ch_start(:) = ' '
     do ijk = 1,NNN
      ch_start(ijk:ijk) = UP_CASE(words(j,1)%ch(ijk:ijk))
     enddo
     if (ch_start(1:NNN) == 'DEFORM') then
         i1 = i1 + 1
         where_in_file(i1) = j
         if (NumberOfWords(j)<8) then
             print*, 'ERROR in file:',trim(nf), ' at line ',j, ' more records needed'
             STOP
         endif
! the units
      predef_ff_deform(i1)%units = unit_type
! name
       NNN = words(j,2)%length
       if (NNN > Max_DEFORM_Name_Len) then
          print*, 'WARNING: The name of the DEFORM defined in file:',trim(nf), 'at line',j,' will be truncated to ',Max_DEFORM_Name_Len,' characters'
          NNN = Max_DEFORM_Name_Len
       endif
       predef_ff_DEFORM(i1)%name(1:NNN) = words(j,2)%ch(1:NNN)

       NNN = words(j,3)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within DEFORM defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_DEFORM(i1)%atom_1_name(1:NNN) = words(j,3)%ch(1:NNN)
       NNN = words(j,4)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within DEFORM defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_DEFORM(i1)%atom_2_name(1:NNN) = words(j,4)%ch(1:NNN)
       NNN = words(j,5)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within DEFORM defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_DEFORM(i1)%atom_3_name(1:NNN) = words(j,5)%ch(1:NNN)
       NNN = words(j,6)%length
       if (NNN > Max_Atom_Name_Len) then
          print*, 'WARNING: The name of the ATOM within DEFORM defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_Atom_Name_Len,' characters'
          NNN = Max_Atom_Name_Len
       endif
       predef_ff_DEFORM(i1)%atom_4_name(1:NNN) = words(j,6)%ch(1:NNN)

       NNN = words(j,7)%length
       do ijk=1,NNN
         ch_start(ijk:ijk) = UP_CASE(words(j,7)%ch(ijk:ijk))
       enddo
       if (NNN > Max_DEFORMStyle_Name_Len) then
          print*, 'WARNING: The name of the StyleDEFORM  within DEFORM defined in file:',trim(nf), 'at line',j,' will be truncated to ',&
          Max_DEFORMStyle_Name_Len,' characters'
          NNN = Max_DEFORMStyle_Name_Len
       endif
       predef_ff_DEFORM(i1)%StyleName(1:NNN) = ch_start(1:NNN)

       call locate_word_in_key(trim(predef_ff_DEFORM(i1)%StyleName(1:NNN)),&
                                predef_ff_DEFORM(i1)%N_params,words(j,:), l_found1,kkk) ! for validation
!       if (trim(predef_ff_DEFORM(i1)%StyleName(1:NNN)) == 'XXXXXXXX') then
!          call attempt_integer_strict(words(j,kkk+1)%length,&
!             words(j,kkk+1)%ch(1:words(j,kkk+1)%length), &
!             predef_ff_DEFORM(i1)%N_params, trim(nf), j)
!          call locate_word_in_key(trim(predef_ff_DEFORM(i1)%StyleName(1:NNN)),&
!                                predef_ff_DEFORM(i1)%N_params+1,words(j,:), l_found1,kkk)
!          kkk = kkk + 1
!       endif
        call get_style_deform(trim(predef_ff_deform(i1)%StyleName(1:NNN)),&
                                predef_ff_deform(i1)%N_params,predef_ff_deform(i1)%style)


       do ijk = kkk+1, kkk+1+predef_ff_DEFORM(i1)%N_params - 1
           call attempt_real_strict(words(j,ijk)%length,&
             words(j,ijk)%ch(1:words(j,ijk)%length), &
             predef_ff_DEFORM(i1)%the_params(ijk-kkk), trim(nf), j)
       enddo

       call set_units_deform(trim(predef_ff_deform(i1)%StyleName),&
                            predef_ff_deform(i1)%units, predef_ff_deform(i1)%N_params,&
                            predef_ff_deform(i1)%the_params( 1:predef_ff_deform(i1)%N_params) )

!print*, 'i1 DEFORMs = ',i1, ' :',trim(predef_ff_DEFORM(i1)%StyleName), ' ' , trim( predef_ff_DEFORM(i1)%atom_1_name(1:NNN)), ' ' , &
!         trim(predef_ff_DEFORM(i1)%atom_2_name(1:NNN)), ' ', &
!         trim(predef_ff_DEFORM(i1)%atom_3_name(1:NNN)), ' ',&
!         trim(predef_ff_DEFORM(i1)%atom_4_name(1:NNN)), ' ',&
!         trim(predef_ff_DEFORM(i1)%StyleName(1:NNN)), &
!         'Nparams style=',predef_ff_DEFORM(i1)%N_params,predef_ff_DEFORM(i1)%style, &
!         'the params=',predef_ff_DEFORM(i1)%the_params(1:predef_ff_DEFORM(i1)%N_params)
!read(*,*)

     endif ! (ch_start(1:NNN) == 'DEFORM') then
  enddo  ! j
 enddo !i = 1,how_many


do i = 1, N_predef_ff_deforms
do j = i+1,N_predef_ff_deforms
   if ( trim(predef_ff_deform(i)%name) == trim(predef_ff_deform(j)%name) ) then
     print*, 'ERROR in file:',trim(nf), ' when define OutOfPlaneDeformation; 2 OutOfPlaneDeformation have the same name; see lines',&
              where_in_file(i),where_in_file(j),&
             ' and either remove one of those lines or change the name of one of the OutOfPlaneDeformations at those 2 lines'
     STOP
   endif
enddo
enddo

end subroutine read_predef_ff_deforms
