module parser
private :: get_vdwprms , initial_paths  ! No longer used here

contains

  include 'read_predef_ff.f90'
  include 'read_type_atoms.f90'
  include 'read_type_bonds.f90'
  include 'read_type_angles.f90'
  include 'read_type_dihedrals.f90'
  include 'read_type_deforms.f90'


  subroutine initial_paths
   use file_names_data, only : path_out,z_A_path,z_M_path,A_path,M_path
   integer i
   i=mk_dir(trim(path_out))
   call message(i,trim(path_out))

   i=mk_dir(trim(z_A_path))
   call message(i,trim(z_A_path))

   i=mk_dir(trim(z_M_path))
   call message(i,trim(z_M_path))

   i=mk_dir(trim(A_path))
   call message(i,trim(A_path))

   i=mk_dir(trim(M_path))
   call message(i,trim(M_path))

   contains
    subroutine message(i,word)
    integer, intent(IN) :: i
    character(*),intent(IN) :: word
     if (i==0) then
      print*,i,' The path ',trim(word),' was created'
     else
      print*,i,' The path ',trim(word),' already exist; '
     endif
    end subroutine message
  end subroutine initial_paths

  subroutine read_force_field(nf)
  use types_module, only : word_type,two_I_one_L_type,two_I_type,one_I_one_L_type
  use allocate_them, only : vdw_type_alloc, atom_type_alloc, mol_type_alloc, connectivity_type_alloc,&
                          interpolate_alloc, pairs_14_alloc , atom_style_alloc
  use paralel_env_data
  use sys_data
  use atom_type_data
  use mol_type_data
  use vdw_type_data
  use connectivity_type_data
  use chars
  use char_constants
  use vdw_def
  use intramol_forces_def
  use Ewald_data
  use sizes_data, only : N_TYPES_DISTRIB_CHARGES 
  use max_sizes_data
  use CTRLs_data
  use field_constrain_data
  use file_names_data, only : MAX_CH_size
  use force_field_data, only : vdw_ff_type, atom_ff_type, &
                        predef_ff_atom,Def_ff_atom,&
                        predef_ff_vdw,Def_ff_vdw,&
                        predef_ff_bond,predef_ff_angle,predef_ff_dih,&
                        N_predef_ff_atoms,N_predef_ff_vdw,&
                        N_predef_ff_bonds,N_predef_ff_angles,N_predef_ff_dihs,&
                        overwrite_2ffvdw,overwrite_2ffatom

  implicit none
  character(*), intent(IN) :: nf
  integer lines, col
  character(1), allocatable :: the_lines(:,:)
  character(1), allocatable :: line(:)
  integer, allocatable :: SizeOfLine(:), NumberOfWords(:)
  integer N,k,i,j,Max_words_per_line, NNN, i_type,iline,i1,iii,jjj,kkk,istart,iend, jj,i2,j1,j2,ii
  type(word_type), allocatable :: the_words(:,:)
  character(250) key, temp_word
  type(two_I_one_L_type) which
  type(one_I_one_L_type) searched_word
  logical, allocatable :: l_skip_line(:), l_do(:), l_id1(:),l_id2(:), l_id22(:,:)
  type(two_I_one_L_type),allocatable ::  where_mol_starts(:), where_mol_ends(:)
  type(two_I_one_L_type),allocatable ::  where_atom_starts(:), where_atom_ends(:)
  type(two_I_one_L_type),allocatable ::  where_bond_starts(:), where_bond_ends(:)
  type(two_I_one_L_type),allocatable ::  where_angle_starts(:), where_angle_ends(:)
  type(two_I_one_L_type),allocatable ::  where_dihedral_starts(:), where_dihedral_ends(:)
  type(two_I_one_L_type),allocatable ::  where_deform_starts(:), where_deform_ends(:)
  type(two_I_one_L_type),allocatable ::  where_geometry_starts(:), where_geometry_ends(:)
  type(two_I_one_L_type),allocatable ::  where_starts(:), where_ends(:)

  integer, allocatable :: kkkv(:), how_many(:), list_how_many(:,:), which_vdw(:,:)
  integer max_prms_intra(25)
  logical l_temp1
  integer ivdw,ivdw1,ivdw2, kk
  character(100) local_text(30)
  integer local_integer(30)

  integer, allocatable :: atom_2_mol(:), angle_2_mol(:),dihedral_2_mol(:)
  integer, allocatable :: angle_2_mol_end(:),dihedral_2_mol_end(:)
  integer, allocatable :: deform_2_mol(:),deform_2_mol_end(:)
  integer, allocatable :: bond_2_mol(:),bond_2_mol_end(:)

   call initial_paths

   call initializations
! get the predefined force field
   call read_predef_ff_dummies(trim(nf),the_words,SizeOfLine,NumberOfWords)
   call read_predef_ff_atoms(trim(nf),the_words,SizeOfLine,NumberOfWords)
   call read_predef_ff_vdw(trim(nf),the_words,SizeOfLine,NumberOfWords)
   call read_predef_ff_bonds(trim(nf),the_words,SizeOfLine,NumberOfWords)
   call read_predef_ff_angles(trim(nf),the_words,SizeOfLine,NumberOfWords)
   call read_predef_ff_dihedrals(trim(nf),the_words,SizeOfLine,NumberOfWords)
   call read_predef_ff_deforms(trim(nf),the_words,SizeOfLine,NumberOfWords)
! \get the predefined force field   
   call scan_for_sizes_lvl_0 ! N_TYPE_MOLECULES
     allocate(where_mol_starts(N_TYPE_MOLECULES),where_mol_ends(N_TYPE_MOLECULES))
     call mol_type_alloc  ! ALLOCATE MOL TYPE SIZES
   call scan_for_sizes_lvl_1 ! How many molecules of each type and their name
   call scan_mol_tags
   N_TYPE_ATOMS = sum(N_type_atoms_per_mol_type)
   MX_excluded = maxval(N_type_atoms_per_mol_type)
    call atom_type_alloc ! ALLOCATE ATOM TYPE SIZES
    call get_atom_type_info
    call interpolate_alloc
   allocate(mol_type_xyz0(N_TYPE_MOLECULES,1:maxval(N_type_atoms_per_mol_type),3)) ! standard orientation
   mol_type_xyz0=0.0d0
   N_type_bonds = sum(Nm_type_bonds(1:N_TYPE_MOLECULES))
   N_type_constrains = sum(Nm_type_constrains(1:N_TYPE_MOLECULES))
   N_type_angles = sum(Nm_type_angles(1:N_TYPE_MOLECULES))
   N_type_dihedrals = sum(Nm_type_dihedrals(1:N_TYPE_MOLECULES))
   N_type_deforms = sum(Nm_type_deforms(1:N_TYPE_MOLECULES))
print*, ' N_type_bonds =',N_type_bonds,N_type_constrains,N_type_angles,N_type_dihedrals,N_type_pairs_14
print*, 'Nm_type_bonds=',Nm_type_bonds
print*, 'Nm_type_constrains=',Nm_type_constrains
print*, 'Nm_type_angles=',Nm_type_angles
print*, 'Nm_type_dihedrals=',Nm_type_dihedrals
print*, 'Nm_type_deforms=',Nm_type_deforms

     call connectivity_type_alloc
     call correct_conectivity_for_rigid_groups ! get the mol_type_xyz0 and constraints


     call get_bond_angle_dihedral_deform_info
print*, 'before remap_and_validate_vdw'
     call remap_and_validate_vdw
print*, 'after remap_and_validate_vdw and before write_data_1'
   call write_data_1
print*, 'after write_data_1'
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


 i1 = 0
 do i = 1, N_TYPE_MOLECULES
   do j = 1, Nm_type_bonds(i)
     i1 = i1 + 1
     bond_types(4,i1) = i
   enddo
 enddo

   ! SYSDEF  call  get_the_total_atoms_and_mols(N_TYPE_MOLECULES,N_mols_of_type,N_type_atoms_per_mol_type,Nmols,Natoms)
!!! ALL DATA IS READ NOW GET THE sysdef.


 if(allocated(atom_2_mol)) deallocate(atom_2_mol)
 if(allocated(bond_2_mol)) deallocate(bond_2_mol)
 if(allocated(bond_2_mol_end)) deallocate(bond_2_mol_end)
 if(allocated(angle_2_mol)) deallocate(angle_2_mol)
 if(allocated(angle_2_mol_end)) deallocate(angle_2_mol_end)
 if(allocated(dihedral_2_mol)) deallocate(dihedral_2_mol)
 if(allocated(dihedral_2_mol_end)) deallocate(dihedral_2_mol_end)

 CONTAINS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine initializations
  call scan_size_of_file(trim(nf),lines, col)
  print*, 'in initialisations lines col=',lines, col
  if (lines==0.or.col==0) then
   print*, 'ERROR : EMPTY ',trim(nf),' FILE; THE PROGRAM WILL STOP'
   stop
  endif
  allocate(the_lines(lines,col),SizeOfLine(lines))
  call get_text_from_file(trim(nf),lines, col,the_lines,SizeOfLine)
  do i=1,lines
   do j = 1, SizeOfLine(i)
      write(6,'(A1)',advance='no') the_lines(i,j)
   enddo
   write(6,*)
  enddo
  allocate(NumberOfWords(lines))
  do k = 1, lines
  call get_number_of_words(SizeOfLine(k), the_lines(k,1:SizeOfLine(k)) ,NumberOfWords(k))
  enddo
  Max_words_per_line = maxval(NumberOfWords)
  allocate(the_words(lines,Max_words_per_line)  )
  do k=1, lines
  call get_the_words(SizeOfLine(k), the_lines(k,1:SizeOfLine(k)), NumberOfWords(k),the_words(k,1:NumberOfWords(k))  )
  enddo
  allocate(l_skip_line(lines)); l_skip_line=.false.;
!  do i = 1, lines ;
!   if (SizeOfLine(i)==0.or.NumberOfWords(i)==0.or.the_lines(i,1)==comment_character_1.or.the_lines(i,1)==comment_character_2) then
!    l_skip_line(i) = .true.
!   endif
!  enddo
end subroutine initializations
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine scan_for_sizes_lvl_0
integer how_many
integer i_start,i_end
integer lines,i,j,k,NNN,i1
integer, allocatable :: where_in_file(:)
character(MAX_CH_SIZE) ch_start

i_start = lbound(the_words,dim=1)
i_end = ubound(the_words,dim=1)
Max_words_per_line= ubound(the_words,dim=2) - lbound(the_words,dim=2) + 1
lines = i_end - i_start + 1
if (allocated(l_skip_line))deallocate(l_skip_line)
allocate(l_skip_line(i_start:i_end)); l_skip_line=.false.

  key='MOLECULAR_TYPES'
  i1 = 0
  do i = i_start,i_end
     NNN = the_words(i,1)%length
     do k = 1, NNN
       ch_start(k:k) = UP_CASE(the_words(i,1)%ch(k:k))
     enddo
     if (ch_start(1:NNN) == trim(key)) i1 = i1 + 1
  enddo

  how_many=i1

  if (how_many > 1) then
    print*, 'ERROR in the input file ',trim(nf), ' MOLECULAR_TYPES must be defined only once and not ',how_many,' times'
    print*, 'delete several MOLECULAR_TYPES records and keep only one'
    STOP
  endif
  if (how_many == 0) then
     print*, 'ERROR in the input file ',trim(nf), ' MOLECULAR_TYPES not defined'
     print*, 'define how many molecular types are in the simulated system'
     STOP
  endif

  call search_words(1,lines,lines,Max_words_per_line,&
                  the_words,SizeOfLine,NumberOfWords,&
                  trim(key),&
                  l_skip_line,which,trim(nf),.true.)
  call get_integer_after_word(trim(nf), which,  trim(key),&
                  NumberOfWords(which%line),the_words(which%line,1:NumberOfWords(which%line)),&
                  N_TYPE_MOLECULES )
print*, 'MOLECULAR TYPES =', N_TYPE_MOLECULES

end subroutine scan_for_sizes_lvl_0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine scan_for_sizes_lvl_1 !How many molecules of each type and their name
! Get the number of molecules of each type and the name of the molecules
  integer how_many
  logical l1,l2,l_1,l_2, l_point,l_found,l_ID
  integer, allocatable :: ff_hit(:)
  integer i1,i2,iii1,iii2
  character(MAX_CH_SIZE) chtemp
  integer i_index

  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'MOLECULE_TYPE',l_skip_line,N_TYPE_MOLECULES,where_mol_starts,.true.)
  do i = 1, ubound(where_mol_starts,dim=1)
  if (where_mol_starts(i)%line==0) then
     print*, 'ERROR in file ',trim(nf),'  Too few MOLECULE_TYPE .. END_MOLECULE_TYPE (fewer than N_TYPE_MOLECULES) '
     print*, 'in scan_for_sizes_lvl_1 where_mol_starts=',where_mol_starts(i)%line
     stop 
  endif
  enddo
  do i = 1, N_TYPE_MOLECULES
  if (NumberOfWords(where_mol_starts(i)%line) < 3) then
  write(6,*) 'ERROR in "',trim(nf),'"file : at line',where_mol_starts%line, ' at least 3 records are needed',&
              'instead of just ',NumberOfWords(where_mol_starts%line)
   STOP
  endif
  enddo
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'END_MOLECULE_TYPE',l_skip_line,N_TYPE_MOLECULES,where_mol_ends,.true.)
  do i = 1, N_TYPE_MOLECULES
  if (where_mol_starts(i)%line >= where_mol_ends(i)%line) then
   write(6,*) 'ERROR in "',trim(nf),'"file : The description of the molecule starts before description ends; ',&
              ' i.e. the key END_MOLECULE_TYPE appear before the key MOLECULE_TYPE ; see line ', where_mol_ends(i)%line
   STOP
  endif
  enddo

include 'atom_types_in_sizes_frg.f90'
include 'bond_types_in_sizes_frg.f90'
include 'angle_types_in_sizes_frg.f90'
include 'dihedral_types_in_sizes_frg.f90'
include 'deform_types_in_sizes_frg.f90'

 l_point=.false.
 do i = 1, N_TYPE_MOLECULES
 iline = where_mol_starts(i)%line
 NNN = the_words(iline,3)%length
 call select_int_real_text_type(NNN,the_words(iline,3)%ch(1:NNN),i_type)
 if (i_type /= 1) then
    write(6,*) 'ERROR in "',trim(nf),'"file : at line ',iline, ' an integer record was expected instrad of: ',&
                the_words(iline,3)%ch(1:NNN)
    STOP
 endif
 call  get_integer_from_string(NNN, the_words(iline,3)%ch(1:NNN), N_mols_of_type(i) )
 do j = 1, min(Max_Mol_Name_Len,the_words(iline,2)%length)
  mol_type_name(i)(j:j) = the_words(iline,2)%ch(j:j)
 enddo
 if (NumberOfWords(iline) > 3) then
  call locate_UPCASE_word_in_key('RIGID_GROUP',0,the_words(iline,:), l_found,kkk)
  l_RIGID_GROUP_TYPE(i) = l_found.and.kkk>3
  call locate_UPCASE_word_in_key('+STAT',0,the_words(iline,:), l_found,kkk)
  if (l_found.and.kkk>3) then
    l_point=.true.
  endif
  call locate_UPCASE_word_in_key('*SFC',0,the_words(iline,:), l_found,kkk)
  is_mol_type_sfc(i) = l_found.and.kkk>3
  if (is_mol_type_sfc(i)) then
    if (NumberofWords(iline) < kkk + 1) then
     print*, 'ERROR in in.in file : One more real record is needed at line :',iline
     STOP
    endif
  NNN = the_words(iline,kkk+1)%length
  call get_real_from_string(NNN, the_words(iline,kkk+1)%ch(1:NNN), param_mol_type_sfc(i) )
  endif 
 endif
 enddo
print*,'is_mol_type_sfc=',is_mol_type_sfc
l_ANY_S_FIELD_CONS_CTRL=.false.
do i = 1, ubound(is_mol_type_sfc,dim=1)
if (is_mol_type_sfc(i))   l_ANY_S_FIELD_CONS_CTRL=.true.
enddo
 
 if (l_point) then
  l_TYPEmol_do_stat_on_type=.false.
  do i = 1, N_TYPE_MOLECULES
  iline = where_mol_starts(i)%line
  if (NumberOfWords(iline) > 3) then
      call locate_UPCASE_word_in_key('+STAT',0,the_words(iline,:), l_found,kkk)
      l_TYPEmol_do_stat_on_type(i) = l_found.and.kkk>3
  endif ! NumberOfWords>3
  enddo
 endif ! l_point

print*, 'statistics of molecule types: ',l_TYPEmol_do_stat_on_type


end subroutine scan_for_sizes_lvl_1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine scan_mol_tags
integer i1,i2,j,k,ijk,ksave1,ksave2,ijksave1,ijksave2,kkk,iline,icount,NNN
logical l_found


  do i =1, N_TYPE_MOLECULES
  iline = where_mol_starts(i)%line
  if (NumberOfWords(iline) > 3) then
  call locate_UPCASE_word_in_key('TAGS',0,the_words(iline,:), l_found,kkk)
  if (l_found.and.kkk>3) then
         i1=0;i2=0
          do ijk = kkk+1, NumberOfWords(iline) 
             do k = 1, the_words(iline,ijk)%length   
             if (the_words(iline,ijk)%ch(k:k) == '(') then 
                i1=i1+1  
                ksave1 = k
                ijksave1 = ijk
                goto 3
             endif
             enddo
          enddo !
3 continue 
          do ijk = kkk+1, NumberOfWords(iline)
             do k = 1, the_words(iline,ijk)%length
             if (the_words(iline,ijk)%ch(k:k) == ')') then
                i2=i2+1
                ksave2 = k
                ijksave2 = ijk
                goto 4
             endif
             enddo
          enddo
4 continue
        if (i1 == 0) then
            print*, 'ERROR in ff.dat file at line ',iline,' after tags you must open a bracket ('
            STOP
        endif
        if (i2 == 0) then
            print*, 'ERROR in ff,dat file at line ',iline,' after tags you must close a bracket )'
            STOP
        endif
        if (ijksave1 > ijksave2 .or. (ijksave1==ijksave2.and.ksave1>ksave2)) then
            print*, 'ERROR in ff.dat file at line ',iline,' after tags the bracket must be oppened first before being closed'
            STOP 
        endif

       icount = 0
       NNN = the_words(iline,ijksave1).length 
       if (NNN > 1 .and. ksave1 /= NNN .and. ijksave1 /= ijksave2) then
         icount = icount + 1
       endif 
       NNN = the_words(iline,ijksave2).length
       if (NNN > 1 .and. ksave2 /= 1 .and. ijksave1 /= ijksave2 ) then
         icount = icount + 1
       endif
       mol_tag(i)%N = icount + ijksave2-ijksave1 - 1 
       if (ijksave1==ijksave2) then 
         if (ksave2==ksave1+1) then 
               mol_tag(i)%N = 0;
         else
               mol_tag(i)%N = 1;
         endif
       endif
       
       if ( mol_tag(i)%N /= 0) then
       allocate(mol_tag(i)%tag( mol_tag(i)%N )); mol_tag(i)%tag(:) = '';
       NNN = the_words(iline,ijksave1).length
         icount = 0
         if (NNN > 1 .and. ksave1 /= NNN .and. ijksave1 /= ijksave2) then
           icount = icount + 1
           mol_tag(i)%tag(icount) = the_words(iline,ijksave1)%ch(ksave1+1:NNN)
         endif
         do ijk = ijksave1+1,ijksave2-1
            icount = icount + 1
            mol_tag(i)%tag(icount) = the_words(iline,ijk)%ch(1:the_words(iline,ijk)%length)
         enddo
       NNN = the_words(iline,ijksave2).length
         if (NNN > 1 .and. ksave2 /= 1 .and. ijksave1 /= ijksave2 ) then
           icount = icount + 1
           mol_tag(i)%tag(icount) = the_words(iline,ijksave2)%ch(1:ksave2-1)
         endif
      
         if (ijksave1==ijksave2) then
             mol_tag(i)%tag(1) = the_words(iline,ijksave2)%ch(ksave1+1:ksave2-1)
         endif
   
       endif !  mol_tag(i)%N /= 0
print*,'molecular tags : tag%N=',mol_tag(i)%N
do j  = 1, mol_tag(i)%N
print*, 'tag=',trim(mol_tag(i)%tag(j))
enddo       
      endif 
    endif
  enddo
print*,'\end molecular tags'
end subroutine scan_mol_tags

!!!!!!!!!!!!!!!!!!!!
subroutine get_atom_type_info
integer i,j,i1,NNN,kkk,jjj,sum_many,i10,i_counter, i_start, i_end, ibla, max_coloumns
integer k1,k2
character(250) key, chtemp, another_file, chtext
character(100) local_text(30)
integer local_integer(30)
logical l_found, l_OK
logical, allocatable :: ff_hit(:), l_skip(:)
integer many(N_TYPE_ATOMS), ijk
integer, allocatable :: where_in_file(:), local_NumberOfWords(:)
type(word_type) , allocatable :: local_words(:,:)
logical l_data_is_in_file,exist
l_data_is_in_file=.false.
i1 = 0
key = 'ATOM_TYPES'
l_ANY_QP_CTRL=.false.
l_ANY_QP_POL_CTRL=.false.
l_ANY_QG_CTRL=.false.
l_ANY_DIPOLE_CTRL=.false.
l_ANY_QG_POL_CTRL=.false.
l_ANY_DIPOLE_POL_CTRL=.false.
l_ANY_S_FIELD_QP_CONS_CTRL=.false.
l_ANY_S_FIELD_QG_CONS_CTRL=.false.
l_ANY_S_FIELD_DIPOLE_CONS_CTRL=.false.
l_ANY_S_FIELD_CONS_CTRL=.false.



sfc_already_defined=.false.
i1=0
do i = 1, N_TYPE_MOLECULES
 do j = 1, N_type_atoms_per_mol_type(i)
 i1 = i1 + 1
 if (is_mol_type_sfc(i)) then
  IS_TYPE_ATOM_FIELD_CONSTRAINED(i1) = .true.
  atom_type_sfield_CONSTR(i1) = param_mol_type_sfc(i)
  sfc_already_defined(i1) = .true.
 endif
 enddo
enddo
print*,'already def=',sfc_already_defined




allocate(ff_hit(N_predef_ff_atoms))
ff_hit=.false.
i1 = 0
do i = 1, N_TYPE_MOLECULES
!  See if the data is in another file
 i_start = where_atom_starts(i)%line+1
 i_end   = where_atom_ends(i)%line-1
 j=where_atom_starts(i)%line+1
 NNN = the_words(j,1)%length
 do k = 1, NNN
  chtemp(k:k) = UP_CASE(the_words(j,1)%ch(k:k))
 enddo
 if (chtemp(1:NNN) == '\IN_FILE') then
    l_data_is_in_file = .true.
    call locate_UPCASE_word_in_key('\IN_FILE',1,the_words(j,:), l_found,kkk) ! validate to have one more entry
    NNN = the_words(j,kkk+1)%length
    another_file(:) = ' '
    another_file(1:NNN) = the_words(j,kkk+1)%ch(1:NNN)
    print*, 'parser MESSAGE: ATOM info for molecule ',mol_type_name(i), ' will be read from the file', trim(another_file(1:NNN) )
    inquire(file=trim(another_file(1:NNN)), exist=exist) 
    if (.not.exist) then
      print*, 'ERROR the file with atom info ',trim(another_file(1:NNN)), ' requested in file ',&
      trim(nf), ' at line ',j, ' does not exist; STOP'
      STOP
    endif
    call get_words_from_file(trim(another_file),lines, max_coloumns, local_NumberOfWords,local_words) 
    i_start = 1
    i_end = i_start + lines - 1
! get them from file
    chtext(:) = ' '
    chtext = trim(another_file)
 else   ! NOT in another file
    if (i_end-i_start+1 ==0) then
     print*, 'ERROR in file ', trim(nf), ' more records needed to define atoms between lines ', i_start, iend
      STOP
    endif
    max_coloumns = Max_words_per_line
    allocate(local_words(i_start:i_end,1:max_coloumns)) ;
    allocate(local_Numberofwords(i_start:i_end))
    do k1 = i_start,i_end
    do k2 = 1,max_coloumns
      local_words(k1,k2)%length = the_words(k1,k2)%length
      local_words(k1,k2)%ch     = the_words(k1,k2)%ch
    enddo
    enddo
    local_Numberofwords(i_start:i_end) = Numberofwords(i_start:iend)
    chtext(:) = ' '
    chtext = trim(nf)
 endif

! \See if the data is in another file
  allocate(where_in_file(i_start:i_end))

  do k = i_start, i_end ; where_in_file(k) = k ; enddo
!print*,atom_type_molindex

  call read_type_atoms(i,i1,i_start, i_end, many, local_NumberOfWords,&
                            max_coloumns, local_words, &
                            ff_hit, where_in_file, trim(chtext))

!print*,atom_type_molindex
 deallocate(local_words)
 deallocate(local_Numberofwords)
 deallocate(where_in_file)
 enddo  ! i
! Now see how many ff I hit and the N_STYLE_ATOM


i_counter = 0
do i = 1, N_predef_ff_atoms
if (ff_hit(i)) then
 i_counter = i_counter + 1
endif
enddo
N_STYLE_ATOMS = i_counter 

print*, 'N_STYLE_ATOMS=',N_STYLE_ATOMS
call atom_Style_alloc
allocate(Def_ff_atom(N_STYLE_ATOMS))
i_counter=0
do i = 1, N_predef_ff_atoms
if (ff_hit(i)) then
 i_counter = i_counter + 1
 call overwrite_2ffatom(Def_ff_atom(i_counter),predef_ff_atom(i))
endif
enddo

deallocate(predef_ff_atom)

i_counter = 0
do i = 1, N_predef_ff_atoms
if (ff_hit(i)) then
 i_counter = i_counter + 1
 do j = 1, N_TYPE_ATOMS
   if(map_atom_type_to_predef_atom_ff(j) == i) then
      map_atom_type_to_style(j) = i_counter
   endif
 enddo
endif
enddo

print*, 'DEFINED ATOMS'
do i = 1, N_STYLE_ATOMS
  print*, i, ' : ',trim(Def_ff_atom(i)%name)
enddo

deallocate(ff_hit)

print*, 'exit from get_atom_type_info'

end subroutine get_atom_type_info

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine get_bond_angle_dihedral_deform_info
integer i,j,i1,NNN,kkk,jjj,sum_many,i10,i_counter, i_start, i_end, ibla, max_coloumns,ivar,kkkk
integer i_bond, i_constrain
integer k1,k2
character(250) key, chtemp, another_file, chtext
character(100) local_text(30)
integer local_integer(30)
logical l_found, l_OK
logical, allocatable :: ff_hit(:), l_skip(:)
integer many(N_TYPE_ATOMS), ijk
integer, allocatable :: where_in_file(:), local_NumberOfWords(:)
type(word_type) , allocatable :: local_words(:,:)
logical l_data_is_in_file,exist
type(two_I_one_L_type),allocatable ::where_local_starts(:), where_local_ends(:)
logical l_proceed
i_bond=0;i_constrain=0

do kkkk = 1, 4 ! kkkk = 1 : bonds ; kkkk = 2 : angles ; kkkk = 3 ; dihedrals ; kkk = 4 out of plane deformstions

l_data_is_in_file=.false.
i1 = 0
key(:) = ' ';
select case (kkkk)
case(1); 
  key = 'BOND_TYPES'
if (N_type_bonds>0)then
  allocate(where_local_starts(lbound(where_bond_starts,dim=1):ubound(where_bond_starts,dim=1)))
  allocate(where_local_ends(lbound(where_bond_starts,dim=1):ubound(where_bond_ends,dim=1)))
endif
case(2); 
if (N_type_angles>0)then
  key = 'ANGLE_TYPES' 
  allocate(where_local_starts(lbound(where_angle_starts,dim=1):ubound(where_angle_starts,dim=1)))
  allocate(where_local_ends(lbound(where_angle_ends,dim=1):ubound(where_angle_ends,dim=1)))
endif
case(3); 
if(N_type_dihedrals>0) then
  key = 'DIHEDRAL_TYPES'
  allocate(where_local_starts(lbound(where_dihedral_starts,dim=1):ubound(where_dihedral_starts,dim=1)))
  allocate(where_local_ends(lbound(where_dihedral_ends,dim=1):ubound(where_dihedral_ends,dim=1)))
endif
case(4)
if (N_type_deforms>0) then 
  key = 'OUTOFPLANEDEFORMS_TYPES'
  allocate(where_local_starts(lbound(where_deform_starts,dim=1):ubound(where_deform_starts,dim=1)))
  allocate(where_local_ends(lbound(where_deform_ends,dim=1):ubound(where_deform_ends,dim=1)))
endif
end select

allocate(ff_hit(N_predef_ff_atoms))
ff_hit=.false.
i1 = 0
do i = 1, N_TYPE_MOLECULES
l_proceed = .false.
select case (kkkk)
case(1); 
if (N_type_bonds>0)then
  ivar = bond_2_mol(i)
  where_local_starts%line = where_bond_starts%line 
  where_local_ends%line = where_bond_ends%line
  l_proceed = .true.
endif
case(2); 
if (N_type_angles>0)then
  ivar = angle_2_mol(i)
  where_local_starts%line = where_angle_starts%line
  where_local_ends%line = where_angle_ends%line
  l_proceed = .true.
endif
case(3);
if(N_type_dihedrals>0) then 
  ivar = dihedral_2_mol(i)
  where_local_starts%line = where_dihedral_starts%line
  where_local_ends%line = where_dihedral_ends%line
  l_proceed = .true.
endif
case(4);
if (N_type_deforms>0) then
  ivar = deform_2_mol(i)
  where_local_starts%line = where_deform_starts%line
  where_local_ends%line = where_deform_ends%line
  l_proceed = .true.
endif
end select

if (l_proceed) then
if (ivar > 0) then
j = ivar
 i_start = where_local_starts(j)%line+1
 i_end   = where_local_ends(j)%line-1
 NNN = the_words(i_start,1)%length
 do k = 1, NNN
  chtemp(k:k) = UP_CASE(the_words(i_start,1)%ch(k:k))
 enddo
 if (chtemp(1:NNN) == '\IN_FILE') then
    l_data_is_in_file = .true.
    call locate_UPCASE_word_in_key('\IN_FILE',1,the_words(i_start,:), l_found,kkk) ! validate to have one more entry
    NNN = the_words(i_start,kkk+1)%length
    another_file(:) = ' '
    another_file(1:NNN) = the_words(i_start,kkk+1)%ch(1:NNN)
    print*, 'parser MESSAGE:',trim(key), 'info for molecule ',mol_type_name(i), ' will be read from the file', trim(another_file(1:NNN) )
    inquire(file=trim(another_file(1:NNN)), exist=exist)
    if (.not.exist) then
      print*, 'ERROR the file with ',trim(key),'  info ',trim(another_file(1:NNN)), ' requested in file ',&
      trim(nf), ' at line ',i_start, ' does not exist; STOP'
      STOP
    endif
    call get_words_from_file(trim(another_file),lines, max_coloumns, local_NumberOfWords,local_words)
    i_start = 1
    i_end = i_start + lines - 1
! get them from file
    chtext(:) = ' '
    chtext = trim(another_file)
 else   ! NOT in another file
    if (i_end-i_start+1 ==0) then
     print*, 'ERROR in file ', trim(nf), ' more records needed to define ',trim(key),' between lines ', i_start, iend
      STOP
    endif
    max_coloumns = Max_words_per_line
    allocate(local_words(i_start:i_end,1:max_coloumns)) ;
    allocate(local_Numberofwords(i_start:i_end))
    do k1 = i_start,i_end
    do k2 = 1,max_coloumns
      local_words(k1,k2)%length = the_words(k1,k2)%length
      local_words(k1,k2)%ch     = the_words(k1,k2)%ch
    enddo
    enddo
    local_Numberofwords(i_start:i_end) = Numberofwords(i_start:iend)
    chtext(:) = ' '
    chtext = trim(nf)
 endif ! ivar

  allocate(where_in_file(i_start:i_end))

  do k = i_start, i_end ; where_in_file(k) = k ; enddo
  select case (kkkk) 
  case(1)
  call read_type_bonds(i, i1,i_bond, i_constrain, bond_2_mol, &
            i_start, i_end, local_NumberOfWords(i_start:i_end),&
             max_coloumns, local_words(i_start:i_end,1:max_coloumns), &
                             where_in_file(i_start:i_end), trim(chtext))

  case(2) 
  call read_type_angles(i, i1, angle_2_mol, &
            i_start, i_end, local_NumberOfWords(i_start:i_end), &
            max_coloumns, local_words(i_start:i_end,1:max_coloumns), &
                             where_in_file(i_start:i_end), trim(chtext))

  case(3)
  call read_type_dihedrals(i, i1, dihedral_2_mol, &
            i_start, i_end, local_NumberOfWords(i_start:i_end),&
             max_coloumns, local_words(i_start:i_end,1:max_coloumns), &
                             where_in_file(i_start:i_end), trim(chtext))
  case(4) 
  call read_type_deforms(i, i1, deform_2_mol, &
            i_start, i_end, local_NumberOfWords(i_start:i_end),&
             max_coloumns, local_words(i_start:i_end,1:max_coloumns), &
                             where_in_file(i_start:i_end), trim(chtext))


  end select

 deallocate(local_words)
 deallocate(local_Numberofwords)
 deallocate(where_in_file)
 endif ! ivar > 0
 endif ! l_proceed
 enddo  ! i

 if (allocated(where_local_starts)) deallocate(where_local_starts)
 if (allocated(where_local_ends)) deallocate(where_local_ends)
 if (allocated(ff_hit)) deallocate(ff_hit)
 enddo ! kkkk


end subroutine get_bond_angle_dihedral_deform_info

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine correct_conectivity_for_rigid_groups
integer i,j,k,iatom,kkk,i1,i2
real(8) sm,t(3)
print*, 'l_RIGID_GROUP_TYPE=',l_RIGID_GROUP_TYPE
do i = 1, N_TYPE_MOLECULES
   if (l_RIGID_GROUP_TYPE(i)) then
     l_skip_line(where_mol_starts(i)%line:where_mol_ends(i)%line)=.false.
     which%find=.false.
       call search_words(where_mol_starts(i)%line,where_mol_ends(i)%line,lines,Max_words_per_line,&
                     the_words(where_mol_starts(i)%line:where_mol_ends(i)%line,1:Max_words_per_line),&
                     SizeOfLine,NumberOfWords(where_mol_starts(i)%line:where_mol_ends(i)%line)&
                     ,'\GEOMETRY',l_skip_line(where_mol_starts(i)%line:where_mol_ends(i)%line),which,trim(nf),.false.,&
                     l_print_warning=.false.)
       if (.not.which%find) then 
         print*, 'ERROR: for a rigid group the Geometry of the molecule must be defined with keyword: \GEOMETRY'
         print*, 'Define geometry in input file between the lines:', where_mol_starts(i)%line,':',where_mol_ends(i)%line
         STOP
       endif
    do j = which%line+1, which%line + N_type_atoms_per_mol_type(i)  
    iatom = j-which%line
      if (NumberOfWords(j) < 4)  then
           print*, 'ERROR in input file at line: ',j,' at least 4 recors are needed (1 integer and 3 reals)'
           STOP
      endif
      do kkk = 2, 4 !note: just ignode the first one 
         NNN = the_words(j,kkk)%length
         call attempt_real_strict(NNN,the_words(j,kkk)%ch(1:NNN), &
                                  mol_type_xyz0(i,iatom,kkk-1), trim(nf), j)
      enddo  ! kkk
    enddo
! RECENTER IT with respect to mass centra.
    t = 0.0d0; sm = 0.0d0
    do j = 1, N_type_atoms_per_mol_type(i)
      i1 = sum(N_type_atoms_per_mol_type(1:i-1))  +  j 
      t(:) = t(:) + mol_type_xyz0(i,j,:)*atom_type_mass(i1)
      sm = sm + atom_type_mass(i1)
    enddo
    t(:) = t(:) / sm 
    do k = 1,3; mol_type_xyz0(i,:,k) = mol_type_xyz0(i,:,k) - t(k); enddo

! REDEFINE THE CONSTRAINS
   do j = 1, Nm_type_constrains(i)
     i1=constrain_types(1,j)
     i2=constrain_types(2,j)
     t(:) = mol_type_xyz0(i,i1,: )-mol_type_xyz0(i,i2,:)
     prm_constrain_types(1,j) = dsqrt(dot_product(t,t)) 
   enddo 
   print*,'parser MESSAGE: THE CONSTRAINST for rigid molecules were re-defined according to molecule geometry'
   endif
  
enddo ! i = 1, N_TYPE_MOLECULES

do i = 1, N_TYPE_MOLECULES
  if (N_type_atoms_per_mol_type(i)==1) then 
    l_RIGID_GROUP_TYPE(i) = .true.
    mol_type_xyz0(i,:,:) = 0.0d0 ! not required but just in case....
  endif
enddo
end subroutine correct_conectivity_for_rigid_groups

subroutine remap_and_validate_vdw
 integer i,j,kkk,NNN,jj,jjj,i1,istyle,jstyle
 logical, allocatable :: l_id1(:),l_id22(:,:)
 logical l,l_1,l_2,l_found
 integer k_found
 integer, allocatable :: which(:,:)

 print*, 'got into remap_and_validate_vdw'


!  allocate(which_vdw_style(N_STYLE_ATOMS,N_STYLE_ATOMS))
!
!    do j=1, N_STYLE_ATOMS
!    do k=1, N_STYLE_ATOMS
!       
!    enddo
!    enddo

     allocate(which(N_STYLE_ATOMS,N_STYLE_ATOMS))
     i1 = 0
     do i = 1, N_STYLE_ATOMS
     do j = i, N_STYLE_ATOMS
         i1 = i1 + 1
         which_atomStyle_pair(i,j) = i1
         which_atomStyle_pair(j,i) = i1
         pair_which_style(i1)%i = i
         pair_which_style(i1)%j = j
     enddo
     enddo

     do i = 1,N_TYPE_ATOMS
       istyle = map_atom_type_to_style(i)
       do j = 1,N_TYPE_ATOMS
       jstyle = map_atom_type_to_style(j)
       l_found=.false.
       do k = 1, N_predef_ff_vdw       
         l_1=trim(atom_type_name(i)) == trim(predef_ff_vdw(k)%atom_1_name).and. &
             trim(atom_type_name(j)) == trim(predef_ff_vdw(k)%atom_2_name)
         l_2=trim(atom_type_name(i)) == trim(predef_ff_vdw(k)%atom_2_name).and. &
             trim(atom_type_name(j)) == trim(predef_ff_vdw(k)%atom_1_name)
         if (l_1.or.l_2) then
           l_found=.true.
           k_found = k
         endif
       enddo         
       if (.not.l_found) then
         print*, 'ERROR : no vdw interaction was defined for atomic pairs ',i,j, ' : ',&
         trim(atom_type_name(i)),'  ',trim(atom_type_name(j))
         STOP
       else
         which_vdw_style(istyle,jstyle) = k_found
         kkk = which_atomStyle_pair(istyle,jstyle)
         atom_Style2_vdwStyle(kkk) = predef_ff_vdw(k_found)%Style 
         atom_Style2_vdwPrm(1:MX_VDW_PRMS,kkk) = predef_ff_vdw(k_found)%the_params(1:MX_VDW_PRMS)
         atom_Style2_N_vdwPrm(kkk) = predef_ff_vdw(k_found)%N_params
       endif
       enddo
     enddo
  atom_Style_1GAUSS_charge_distrib(:) = 1.0d0/Def_ff_atom(:)%QGaussWidth
  atom_style_name(:) = ' ' 
  atom_style_name(:) = Def_ff_atom(:)%name
 print*, 'atom_style_name=',atom_style_name


 do i = 1, N_TYPE_ATOMS
   istyle = map_atom_type_to_style(i)
   atom_Style_dipole_pol(iStyle) = atom_type_dipole_pol(i)  
   is_Style_dipole_pol(iStyle)  = is_type_dipole_pol(i)
 enddo 

end subroutine remap_and_validate_vdw


 subroutine write_data_1
   integer i,j,k,kkk,iii,jjj

   write(6,*) 'N_TYPE_ATOMS=',N_TYPE_ATOMS
   write(6,*) 'q  dipol mass name isDummy? which_mol'
   do i = 1, N_TYPE_ATOMS
     write(6,'(2(F8.4,1X), F8.3,1X, A4,1X, L2,1X, I3)'  ) atom_type_charge(i),atom_type_dipol(i),&
                                                      atom_type_mass(i),atom_type_name(i),atom_type_mass(i)==0.0d0!, &
                                                      !atom_type_in_which_mol_type(i)
   enddo
   write(6,*) 'Intramolecular types 1) mol  : bond: constrain: angle: dihedral: deform 14'
   do i = 1, N_TYPE_MOLECULES
     write(6,*)i, Nm_type_bonds(i),Nm_type_constrains(i),Nm_type_angles(i),Nm_type_dihedrals(i),Nm_type_deforms(i),Nm_type_14(i)
   enddo

   ! SYSDEF  call  get_the_total_atoms_and_mols(N_TYPE_MOLECULES,N_mols_of_type,N_type_atoms_per_mol_type,Nmols,Natoms)

   do i = 1, N_TYPE_MOLECULES
     write(6,*) ' MOLNAME=',trim(mol_type_name(i)), '      :NumMols=',N_mols_of_type(i),&
                '  : Atoms-type=',N_type_atoms_per_mol_type(i)
   enddo
!   do i = 1, N_TYPES_VDW
!     write(6,*) i,atom_type_name(vdw_type_atom_1_type(i,1:size_vdw_type_atom_1_type(i))),&
!     atom_type_name(vdw_type_atom_2_type(i,1:size_vdw_type_atom_1_type(i))),&
!     'prm_vdw=', prm_vdw_type(1:vdw_type_Nparams(i),i)
!   enddo

   write(6,*) 'pair  | Style | N_vdwPrm | vdwPrm '
   do i = 1, N_STYLE_ATOMS*(N_STYLE_ATOMS+1)/2
     write(6,'(I3,2(A4,1X),I4,1X,A3,I3,A3)',advance='NO') i, &
     atom_style_name(pair_which_style(i)%i), atom_style_name(pair_which_style(i)%j),&
     atom_style2_vdwStyle(i),' | ', atom_style2_N_vdwPrm(i),' | '
     do j = 1, atom_style2_N_vdwPrm(i)
       write(6,'(1X,F14.5)',advance='no') atom_style2_vdwPrm(j,i)
     enddo 
     write(6,*)
   enddo

 end subroutine write_data_1

end subroutine read_force_field


!-------------------------------------

subroutine read_input_file(nf)

 use cut_off_data
 use Ewald_data
 use ensamble_data
 use ensamble_def_module
 use ewald_def_module
 use types_module, only : word_type,two_I_one_L_type,two_I_type,one_I_one_L_type
 use char_constants
 use chars
 use profiles_data, only : l_need_2nd_profile, l_need_1st_profile, l_1st_profile_CTRL,l_2nd_profile_CTRL,&
                           N_BINS_XX,N_BINS_YY,N_BINS_ZZ, N_BINS_ZZs
 use collect_data
 use integrate_data
 use cg_buffer, only : CG_TOLERANCE, order_lsq_cg_predictor, cg_predict_restart_CTRL,aspc_coars_Niters, &
                       aspc_update_4full_iters, aspc_omega, cg_skip_fourier
 use CTRLs_data
 use sim_cel_data, only : i_boundary_CTRL
 use connectivity_ALL_data, only :  l_exclude_14_dih_CTRL,l_exclude_14_all_CTRL
 implicit none
 character(*), intent(IN) :: nf
 integer iv(100),ivi(100),Nini,NNN
 real(8) v(100)
 integer i,j,k
 logical, allocatable :: l_skip_line(:)
 integer lines, col
 character(1), allocatable :: the_lines(:,:)
 character(1), allocatable :: line(:)
 integer, allocatable :: SizeOfLine(:), NumberOfWords(:)
 integer Max_words_per_line
 type(word_type), allocatable :: the_words(:,:)
 type(two_I_one_L_type) which
 type(one_I_one_L_type) searched_word
 logical l_found


 call initializations

 call search_words(1,lines,lines,Max_words_per_line,&
                  the_words,SizeOfLine,NumberOfWords,&
                  'DO_QN_DYNAMICS',l_skip_line,which,trim(nf),.false.)
 if (which%find) l_do_QN_CTRL = .true.

 Nini = 2 ; ivi(1:Nini) = (/ 2,2 /)
 call get_from_key('CUT_OFF',Nini,ivi,iv(1:Nini),v(1:Nini),l_found)
      cut_off = v(1) ; displacement = v(2)
 print*, 'cut_off disp = ',cut_off,displacement

 Nini = 1 ; ivi(1:Nini) = (/ 2 /)
 preconditioner_cut_off = 4.0d0 ! default value
 call get_from_key('PRECONDITIONER_CUT_OFF',Nini,ivi,iv(1:Nini),v(1:Nini),l_found,.false.)
      if (l_found) preconditioner_cut_off = v(1) ;
 print*, 'preconditioner_cut_off=',preconditioner_cut_off
 Nini = 1 ; ivi(1:Nini) = (/ 2 /)
 call get_from_key('CG_TOLERANCE',Nini,ivi,iv(1:Nini),v(1:Nini),l_found,.false.)
 if (l_found) CG_TOLERANCE =v(1)
 print*,'Conjugate gradient tolerance=',CG_TOLERANCE
 call get_from_CG_TOLERANCE

 call get_Ewald_method(i_type_EWALD_CTRL)
 print*, 'EWALD METHOD = ',i_type_EWALD_CTRL
 if (i_type_EWALD_CTRL /= 0) then
  Nini = 1 ; ivi(1:Nini) = (/ 2 /)
      ewald_alpha = 0.25d0
  call get_from_key('EWALD_ALPHA',Nini,ivi,iv(1:Nini),v(1:Nini),l_found,.false.)
       if (l_found) ewald_alpha = v(1) ;
 print*, 'ewald_alpha=',ewald_alpha
 endif



! I need to see the symetry of the box.
 call get_type_box_in_config
! \\\\I need to see the symetry of the box (from config)


 if (i_type_EWALD_CTRL == 1 .and. i_boundary_CTRL /= 1 ) then ! 3D-SMPE
  Nini = 6 ; ivi(1:Nini) = (/ 1,1,1, 1,1,1 /)
  call get_from_key('EWALD_SMPE_SPLINES_AND_FFT',Nini,ivi,iv(1:Nini),v(1:Nini),l_found)
      order_spline_xx = iv(1) ;order_spline_yy = iv(2) ; order_spline_zz = iv(3)
      nfftx = iv(4) ; nffty = iv(5) ; nfftz = iv(6)
  print*, 'Ewald case: ',i_type_EWALD_CTRL,i_boundary_CTRL
  print*,'osplines ntffts=',order_spline_xx,order_spline_yy,order_spline_zz,nfftx,nffty,nfftz
 endif

 if (i_type_EWALD_CTRL == 1.and. i_boundary_CTRL == 1 ) then ! 2D-SMPE
  Nini = 6 ; ivi(1:Nini) = (/ 1,1,1, 1,1,1 /)
  call get_from_key('EWALD_SMPE_SPLINES_AND_FFT',Nini,ivi,iv(1:Nini),v(1:Nini),l_found)
      order_spline_xx = iv(1) ;order_spline_yy = iv(2) ; order_spline_zz = iv(3)
      nfftx = iv(4) ; nffty = iv(5) ; nfftz = iv(6)
  Nini = 2 ; ivi(1:Nini) = (/ 1,1 /) 
  call get_from_key('EWALD_2D_K0_SPLINES',Nini,ivi,iv(1:Nini),v(1:Nini),l_found)
      order_spline_zz_k0 = iv(1)
      n_grid_zz_k0 = iv(2)
  print*, 'Ewald case ',i_type_EWALD_CTRL,i_boundary_CTRL
  print*, 'osplines ntffts=',order_spline_xx,order_spline_yy,order_spline_zz,nfftx,nffty,nfftz
  print*,'osplines ntffts at k0=',order_spline_zz_k0,n_grid_zz_k0
 endif
 
 if (i_type_EWALD_CTRL == 2) then
  Nini = 3 ; ivi(1:Nini) = (/ 1,1,1 /)
  call get_from_key('EWALD_K_VECTORS',Nini,ivi,iv(1:Nini),v(1:Nini),l_found)
      k_max_x = iv(1) ; k_max_y = iv(2) ; k_max_z = iv(3)
 print*, 'Ewald case ',i_type_EWALD_CTRL,i_boundary_CTRL
 print*, 'k_max_x k_max_y k_max_z=',k_max_x,k_max_y,k_max_z
 endif 

 if (i_type_EWALD_CTRL==2 .and. i_boundary_CTRL == 1 ) then
   Nini = 1 ; ivi(1:Nini) = (/ 2 /)
  call get_from_key('EWALD_2D_Z_SIZE',Nini,ivi,iv(1:Nini),v(1:Nini),l_found)
      h_cut_off2D = v(1)
  print*, 'again  Ewald case ',i_type_EWALD_CTRL,i_boundary_CTRL
  print*, 'EWALD_2D_Z_SIZE=',h_cut_off2D
 endif  


 call get_profiles

 call get_type_ensamble(i_type_ensamble_CTRL,i_type_thermostat_CTRL,i_type_barostat_CTRL)
 print*, 'i type ensamble =',i_type_ensamble_CTRL,i_type_thermostat_CTRL,i_type_barostat_CTRL
 print*, 'thermocopling=',thermo_coupling
 print*,'baro coupling=',barostat_coupling
!stop
print*, 'goto get_excluded_14_flags'
 call get_excluded_14_flags

 Nini = 2 ; ivi(1:Nini) = (/ 1,1 /)
  call get_from_key('COLLECT_STATISTICS',Nini,ivi,iv(1:Nini),v(1:Nini),l_found)
      collect_skip = iv(1)
      collect_length = iv(2)
      if (mod(collect_length,collect_skip) /=0) then
        print*, 'ERROR in input file when defining COLLECT_STATISTICS; collect_skip must divide exctly by collect_length'
        STOP
      endif
 print*, 'COLLECT_STATISTICS=',collect_skip,collect_length
 
 Nini= 1 ; ivi(1:Nini) = (/ 2 /)
  call get_from_key('TIME_STEP',Nini,ivi,iv(1:Nini),v(1:Nini),l_found)
      time_step = v(1)
  print*, 'TIME_STEP=',TIME_STEP

  call get_N_MD_STEPS

Nini = 1 ; ivi(1:Nini) = (/ 2 /)
 dens_var = 1.0d0 ! default value
 call get_from_key('DENS_VAR',Nini,ivi,iv(1:Nini),v(1:Nini),l_found,.false.)
      if (l_found) dens_var = v(1) ;
 print*, 'dens_var=',dens_var

  call CG_details
  call cg_skip
  call get_rsmd
  call get_rdfs 

  call get_lucretius_integrator 

  call get_A_THOLE

  call get_system_preparation
  call get_shake_tol
  call get_quick_stat_preview
  call get_history
  call get_put_CM_molecules_in_box
  call get_temperature_annealing

  call parse_extra_line

contains
include 'get_type_box_in_config.f90'
subroutine initializations
  integer NNN
  call scan_size_of_file(trim(nf),lines, col)
  print*, 'in initialisations lines col=',lines, col
  if (lines==0.or.col==0) then
   print*, 'ERROR : EMPTY ',trim(nf),' FILE; THE PROGRAM WILL STOP'
   stop
  endif
  allocate(the_lines(lines,col),SizeOfLine(lines))
  call get_text_from_file(trim(nf),lines, col,the_lines,SizeOfLine)
  do i=1,lines
   do j = 1, SizeOfLine(i)
      write(6,'(A1)',advance='no') the_lines(i,j)
   enddo
   write(6,*)
  enddo
  allocate(NumberOfWords(lines))
  do k = 1, lines
  call get_number_of_words(SizeOfLine(k), the_lines(k,1:SizeOfLine(k)) ,NumberOfWords(k))
  enddo
  Max_words_per_line = maxval(NumberOfWords)
  allocate(the_words(lines,Max_words_per_line)  )
  do k=1, lines
  call get_the_words(SizeOfLine(k), the_lines(k,1:SizeOfLine(k)), NumberOfWords(k),the_words(k,1:NumberOfWords(k))  )
  enddo
  if (allocated(l_skip_line)) deallocate(l_skip_line)
  allocate(l_skip_line(lines)); l_skip_line=.false.;
  do i = 1, lines ;
   if (SizeOfLine(i)==0.or.NumberOfWords(i)==0.or.the_lines(i,1)==comment_character_1.or.the_lines(i,1)==comment_character_2) then
    l_skip_line(i) = .true.
   endif
  enddo
!print*, 'exit from initialization'
 do i = 1, lines
  if(.not.l_skip_line(i)) then
  NNN = the_words(i,1)%length
  do j = 1, NNN
    the_words(i,1)%ch(j:j) = up_case(the_words(i,1)%ch(j:j))
  enddo
 endif
 enddo
end subroutine initializations

subroutine get_from_key(key,valid_length,itype,iv,v,l_found,l_strict)
 character(*), intent(IN) :: key
 integer, intent(IN) :: valid_length
 integer, intent(IN) :: itype(valid_length)
 logical, intent(IN),optional :: l_strict !(if l_strict = true will stop if the key is not found)
 integer, intent(OUT)::  iv(valid_length)
 real(8), intent(OUT)::  v(valid_length)
 logical, intent(OUT) :: l_found
 integer NNN,i,j,k,kkk,iline
 logical ll

 ll = .true.
 if (present(l_strict)) ll = l_strict

!    key = 'CUT_OFF'
!print*, 'enteringg in get_from_key'
    call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,trim(key),l_skip_line,which,trim(nf),ll)
    l_found=which%find
!    the_words(which%line,kkk)%ch(1:NNN)
    if (ll==.false.) then ! if ll = .true. the program will stop in search_words
      if (.not.which%find) then
        print*, 'WARNING : NO KEY "',trim(key),'" was defined ; a default value will be used'
        RETURN
      endif
    endif

    iline = which%line
    v = 0.0d0 ; iv = 0.0d0
    if  (NumberOfWords(which%line) < valid_length+1) then
      print*, 'ERROR : insuficient records in the file "',trim(nf),'" at line',which%line
      STOP
    endif 
    do i = 1, valid_length
     kkk = i + 1
     NNN = the_words(iline,kkk)%length
      if (itype(i) == 1 ) then
       call attempt_integer_strict(NNN,the_words(iline,kkk)%ch(1:NNN), &
                                         iv(i), nf, iline)        
      elseif (itype(i) == 2) then
       call attempt_real_strict(NNN,the_words(iline,kkk)%ch(1:NNN), &
                                         v(i), nf, iline)
      else
       print*, 'ERROR in get_from_key from read_input_file ; itype can only be 1 or 2 (integer or reals)'
       print*, 'call the subroutine get_from_key with parameter itype to be 1 or 2 only'
       STOP
      endif 
    enddo
!print*, 'in read_input get_key key=',trim(key),'v=',v,'iv=',iv
end subroutine get_from_key

subroutine get_N_MD_STEPS
 use integrate_data, only : N_MD_STEPS,N_MD_STEPS_BLANK_FIRST
 integer kkk,NNN,i_position
 logical l_found
 
     call search_words(1,lines,lines,Max_words_per_line,the_words,&
                SizeOfLine,NumberOfWords,'N_MD_STEPS',l_skip_line,which,trim(nf),.false.)
     if (.not.which%find) then
      print*,'ERROR in file ',trim( nf), ' missing keyword N_MD_STEPS: The number of simulation steps ',&
             'must be defined in input file using N_MD_STEPS <integer number>',&
             'Define N_MD_STEPS in ', trim(nf), ' file and re-run'
      STOP
     endif
     
     kkk=which%line
     if (NumberOfWords(kkk) < 2) then
       print*,'ERROR: More records needed in file ', trim(nf), 'at line ', kkk, ' when defining N_MD_STEPS'
      STOP
     endif
     i_position = 2
     NNN = the_words(kkk,i_position)%length
     call attempt_integer_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                           N_MD_STEPS  , trim(nf), kkk)
                           
     call locate_UPCASE_word_in_key('BLANK_FIRST',1, the_words(which%line,:),l_found,i_position)
     if (l_found) then
       i_position=i_position+1
       NNN = the_words(kkk,i_position)%length
       call attempt_integer_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                          N_MD_STEPS_BLANK_FIRST  , trim(nf), kkk)
     endif
print*,'N_MD_STEPS N_MD_STEPS_BLANK_FIRST =',N_MD_STEPS,N_MD_STEPS_BLANK_FIRST
 end subroutine get_N_MD_STEPS    
subroutine get_Ewald_method(i_type_Ewald)
 integer, intent(OUT) :: i_type_Ewald
 integer NNN
 integer iline
 call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'EWALD_METHOD',l_skip_line,which,trim(nf),.false.)
 if (.not.which%find) then
        print*, 'WARNING : NON EWALD METHOD REQUESTED'
        i_type_Ewald = 0
        RETURN
 endif
 if (NumberOfWords(which%line) < 2) then
       print*, 'WARNING : NON EWALD METHOD REQUESTED'
        i_type_Ewald = 0
        RETURN
 endif
 iline = which%line
 NNN = the_words(iline,2)%length
 call ewald_def(trim(the_words(iline,2)%ch(1:NNN)), i_type_Ewald) 
 
 if (i_type_Ewald==0) then
   print*, 'ERROR in the file "',trim(nf),'"at line: ',which%line,' when defining the Ewald method'
   print*,'The method was not specified or not properly specified'
   STOP
 end if
 end subroutine get_Ewald_method

!-----------

subroutine get_from_CG_TOLERANCE
 use thermostat_Lucretius_data
 use chars, only : w_WP_CASE
 use cg_buffer, only : picard_dumping,l_try_picard_CTRL
 integer kkk,iline,NNN,i
 logical l_skip(1:lines)

 l_skip=.false.

 call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'CG_TOLERANCE',l_skip,which,trim(nf),.false.)

 if (.not.which%find) RETURN

 iline = which%line
 call locate_UPCASE_word_in_key('TRY_PICARD',0,the_words(iline,:), l_found,kkk)
 if (l_found)  l_try_picard_CTRL = .true.
 call locate_UPCASE_word_in_key('PICARD_DUMPING',1,the_words(iline,:), l_found,kkk)
 if (l_found)  &
     call attempt_real_strict(the_words(iline,kkk+1)%length,&
             the_words(iline,kkk+1)%ch(1:the_words(iline,kkk+1)%length), &
             picard_dumping  , trim(nf), iline)

print*, 'l_try_picard_CTRL picard_dumping=',l_try_picard_CTRL,picard_dumping

  
end subroutine get_from_CG_TOLERANCE

subroutine get_lucretius_integrator
 use thermostat_Lucretius_data
 use chars, only : w_WP_CASE
 integer kkk,iline,NNN,i
 logical l_found, l_found1,l_found2,l_found3,l_found4

 call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'INTEGRATOR',l_skip_line,which,trim(nf),.false.)
                 
 
 if (.not.which%find) RETURN
 
 iline = which%line
 if (NumberOfwords(iline)<2) then 
     print*, 'ERROR: more records needed in file ', trim(nf), ' at line ', iline
     STOP
 endif 

 do i = 1, NumberOfwords(iline)
 call w_WP_CASE(the_words(iline,i)%ch(1:the_words(iline,i)%length))
 enddo
 
 kkk = 2; NNN = the_words(iline,kkk)%length
 
 !-----------------------------LUCRETIUS MULTI STEP
 if (the_words(iline,kkk)%ch(1:NNN)=='LUCRETIUS') then
    use_Lucretius_integrator = .true.
    i_type_integrator_CTRL = 999
    call locate_UPCASE_word_in_key('MULTIMED',1,the_words(iline,:), l_found,kkk)
    call attempt_integer_strict(the_words(iline,kkk+1)%length,&
             the_words(iline,kkk+1)%ch(1:the_words(iline,kkk+1)%length), &
             Multi_Med  , trim(nf), iline)
    call locate_UPCASE_word_in_key('MULTIBIG',1,the_words(iline,:), l_found,kkk)
    call attempt_integer_strict(the_words(iline,kkk+1)%length,&
             the_words(iline,kkk+1)%ch(1:the_words(iline,kkk+1)%length), &
             Multi_Big  , trim(nf), iline) 
    call locate_UPCASE_word_in_key('NNOS',1,the_words(iline,:), l_found,kkk)
    call attempt_integer_strict(the_words(iline,kkk+1)%length,&
             the_words(iline,kkk+1)%ch(1:the_words(iline,kkk+1)%length), &
             N_N_O_S  , trim(nf), iline)
    call locate_UPCASE_word_in_key('CUT_SHORT',1,the_words(iline,:), l_found,kkk)
    call attempt_real_strict(the_words(iline,kkk+1)%length,&
             the_words(iline,kkk+1)%ch(1:the_words(iline,kkk+1)%length), &
             cut_off_short  , trim(nf), iline)
    cut_off_short_sq = cut_off_short*cut_off_short         
    call locate_UPCASE_word_in_key('MORE_SPEED',1,the_words(iline,:), l_found,kkk)
    if (l_found) then
    call attempt_integer_strict(the_words(iline,kkk+1)%length,&
             the_words(iline,kkk+1)%ch(1:the_words(iline,kkk+1)%length), &
             lucretius_integrator_more_speed_skip  , trim(nf), iline)
    lucretius_integrator_more_speed_doit = l_found
    if (lucretius_integrator_more_speed_skip<0) then 
       print*,'ERROR in in.in file at line ',iline, ' lucretius_integrator_more_speed_skip must be positive'
       STOP
    endif
    if (lucretius_integrator_more_speed_doit) then
       print*, 'WARNING ; the multistep integrator will insert an additional step which consists in a full ',&
               'evaluation of truncated and shifted forces over the entire cut-off: ! For now dipoles are not included in this scheme'
       print*, 'At any ',lucretius_integrator_more_speed_skip,' timesteps a full update of forces (Ewald dipoles etc) is performed'
    endif
    endif ! lfound MORE_SPEED

    print*, 'Lucretius integrator MultiMed MultiBig NNOS Cut_Short=',Multi_Med,Multi_Big,N_N_O_S,cut_off_short
    if (mod(Multi_Big,Multi_Med) /=0) then
      print*, 'ERROR in file ',trim(nf),' at line ',iline, ' when define lucretius integrator '
      print*, 'MULTIBIG must divide exactly to MULTIMED (mod(MULTIBIG,MULTIMED)=0)'
      STOP
    endif
 else if ( the_words(which%line,kkk)%ch(1:NNN) == 'VV' ) then
      i_type_integrator_CTRL  = 0  ! by dedfault is VV
 else if ( the_words(which%line,kkk)%ch(1:NNN) == 'GEAR_4' ) then
      i_type_integrator_CTRL = 1 ! gear 4
 else
      print*, 'ERROR: in input file at line :',which%line,' Undefined integrator'
      STOP
 endif

  call locate_UPCASE_word_in_key('MASSIVE_ATOM_XYZ',1,the_words(iline,:), l_found,kkk)
  if (l_found) then
    i_type_thermostat_CTRL=-99
    if (N_N_O_S==1) then
     print*,'WARNING THE NUMBER OF NOSE HOOVER CHAINS WAS SET TO N_N_O_S=5'
     N_N_O_S=5
    endif
  endif
   call locate_UPCASE_word_in_key('MASSIVE_ATOM',1,the_words(iline,:), l_found1,kkk)
   if (l_found1) then
   if (l_found) then 
     print*,'ERROR in input file at line ',which%line,'Delete one of the records MASSIVE_ATOM_XYZ or MASSIVE_ATOM'
     STOP
   endif 
    i_type_thermostat_CTRL=-98
    if (N_N_O_S==1) then
     print*,'WARNING THE NUMBER OF NOSE HOOVER CHAINS WAS SET TO N_N_O_S=5'
     N_N_O_S=5
    endif
   endif
   call locate_UPCASE_word_in_key('MASSIVE_MOL_XYZ',1,the_words(iline,:), l_found2,kkk)
   if (l_found2) then
       if (l_found) then
         print*,'ERROR in input file at line ',which%line,'Delete one of the records MASSIVE_ATOM_XYZ or MASSIVE_MOL_XYZ'
         STOP
       endif
       if (l_found1) then
         print*,'ERROR in input file at line ',which%line,'Delete one of the records MASSIVE_ATOM or MASSIVE_MOL_XYZ'
         STOP
       endif
    i_type_thermostat_CTRL=-97
    if (N_N_O_S==1) then
     print*,'WARNING THE NUMBER OF NOSE HOOVER CHAINS WAS SET TO N_N_O_S=5'
     N_N_O_S=5
    endif
   endif

   call locate_UPCASE_word_in_key('MASSIVE_MOL',1,the_words(iline,:), l_found3,kkk)
   if (l_found3) then
       if (l_found) then
         print*,'ERROR in input file at line ',which%line,'Delete one of the records MASSIVE_ATOM_XYZ or MASSIVE_MOL'
         STOP
       endif
       if (l_found1) then
         print*,'ERROR in input file at line ',which%line,'Delete one of the records MASSIVE_ATOM or MASSIVE_MOL'
         STOP
       endif
       if (l_found2) then
         print*,'ERROR in input file at line ',which%line,'Delete one of the records MASSIVE_MOL_XYZ or MASSIVE_MOL'
         STOP
       endif
    i_type_thermostat_CTRL=-96
    if (N_N_O_S==1) then
     print*,'WARNING THE NUMBER OF NOSE HOOVER CHAINS WAS SET TO N_N_O_S=5'
     N_N_O_S=5
    endif
   endif


print*,'i_type_integrator_CTRL NNOS=' , i_type_integrator_CTRL,N_N_O_S,i_type_thermostat_CTRL

end subroutine get_lucretius_integrator
!------------ 

subroutine get_A_THOLE
 use chars, only : w_WP_CASE
 use thole_data
 integer kkk,iline,NNN,i

 call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'A_THOLE',l_skip_line,which,trim(nf),.false.)
 
 if (which%find) then
    iline = which%line
     if (NumberOfwords(iline)<2) then
       print*, 'ERROR: more records needed in file ', trim(nf), ' at line ', iline
       STOP
     endif
     kkk=1
     call attempt_real_strict(the_words(iline,kkk+1)%length,&
             the_words(iline,kkk+1)%ch(1:the_words(iline,kkk+1)%length), &
             aa_thole  , trim(nf), iline)
   print*, 'DEFALUT value of Thole constant modified to ',aa_thole
 endif 
end subroutine get_A_THOLE

subroutine cg_skip
 use cg_buffer, only : l_DO_CG_CTRL_Q,l_DO_CG_CTRL_DIP,l_DO_CG_CTRL, cg_skip_MAIN,&
                       l_do_FFT_in_inner_CG
 use CTRLs_data, only : l_skip_cg_in_first_step_CTRL
   integer NNN
   integer iline,i,N,isave,kkk
   cg_skip_MAIN%Q = 0; cg_skip_MAIN%dip = 0
   l_DO_CG_CTRL_Q=.true.
   l_DO_CG_CTRL_DIP=.true.
   l_DO_CG_CTRL = .true.

   call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'CG_SKIP',l_skip_line,which,trim(nf),.false.)
   if (.not.which%find) RETURN  ! use default values

   call locate_UPCASE_word_in_key('SKIP_CG_FIRST_ITERATION',0,the_words(which%line,:), l_found,kkk)
   if (l_found) then
     print*,'MESSAGE: parser: cg iterations will be skipped in the very first iteration'
     l_skip_cg_in_first_step_CTRL=.true.
   else
     l_skip_cg_in_first_step_CTRL=.false.
   endif 
print*,'l_skip_cg_in_first_step_CTRL=',l_skip_cg_in_first_step_CTRL

   call locate_UPCASE_word_in_key('Q',1,the_words(which%line,:), l_found,kkk)  
   if (l_found) then
     NNN=the_words(which%line,kkk+1)%length
     call attempt_integer_strict(NNN,the_words(which%line,kkk+1)%ch(1:NNN), &
                                cg_skip_MAIN%Q   , trim(nf), which%line)
   else
     cg_skip_MAIN%Q=0
   endif
   call locate_UPCASE_word_in_key('DIPOLE',1,the_words(which%line,:), l_found,kkk)
   if (l_found) then
     NNN=the_words(which%line,kkk+1)%length
     call attempt_integer_strict(NNN,the_words(which%line,kkk+1)%ch(1:NNN), &
                                cg_skip_MAIN%dip   , trim(nf), which%line)
   else
      cg_skip_MAIN%dip = 0 
   endif
!NO_FFT_inner_CG

   call locate_UPCASE_word_in_key('NO_FFT_INNER_CG',0,the_words(which%line,:), l_found,kkk)
   if (l_found) then
     l_do_FFT_in_inner_CG = .false.
   else
     l_do_FFT_in_inner_CG = .true.
   endif

   l_DO_CG_CTRL_Q=.false.
   l_DO_CG_CTRL_DIP=.false.
   l_DO_CG_CTRL = .false.

   if (cg_skip_MAIN%Q ==0) l_DO_CG_CTRL_Q = .true.
   if (cg_skip_MAIN%dip ==0)  l_DO_CG_CTRL_DIP = .true.
   
   print*, 'cg_skip Q DIP = ',cg_skip_MAIN%Q,cg_skip_MAIN%DIP 
   print*, 'do FFT in inner CG?', l_do_FFT_in_inner_CG

end subroutine cg_skip

 
 subroutine CG_details
 
   integer NNN
   integer iline,i,N,isave,kkk
   call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'CG_PREDICTOR',l_skip_line,which,trim(nf),.false.)
   if (.not.which%find) RETURN  ! use default values
   iline = which%line
   if (NumberOfWords(which%line) < 2) then
     PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line, 'AT least one more record needed'
     STOP
   endif
    cg_skip_Fourier%lskip = .false.
    cg_skip_Fourier%how_often=1000000000 ! never
    call locate_UPCASE_word_in_key('SKIPFOURIER',1,the_words(iline,:), l_found,kkk)
    cg_skip_Fourier%lskip=l_found
    if (l_found)  then
         call attempt_integer_strict(the_words(iline,kkk+1)%length,&
             the_words(iline,kkk+1)%ch(1:the_words(iline,kkk+1)%length), &
             cg_skip_Fourier%how_often  , trim(nf), iline)
         print*, 'WARNING: FOURIER PART WILL BE SKIPPED WHEN EVALUATER CHARGES/DIPOLES'
    endif

    NNN = the_words(iline,2)%length
    cg_predict_restart_CTRL=-999
    if (the_words(iline,2)%ch(1:NNN) == 'FROM-ZERO') then
        cg_predict_restart_CTRL = -1
    else if (the_words(iline,2)%ch(1:NNN) == 'LAST-ITER') then
        cg_predict_restart_CTRL = 0 ! just take the result from the last iteration
    else if (the_words(iline,2)%ch(1:NNN) == 'POLY5') then
        cg_predict_restart_CTRL = 1 ! take a polynomial scheme of order 5
    else if (the_words(iline,2)%ch(1:NNN) == 'LEAST-SQUARE') then
      cg_predict_restart_CTRL = 2
      if (cg_skip_Fourier%lskip) then
         N=3+2
      else
         N=3
      endif
      if (NumberOfWords(which%line) < N) then
        PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line, 'AT least one more record needed',&
                 ' to define the order of least square interpolation; recomended value: integer between 5-10'
        STOP
      endif
      NNN=the_words(iline,3)%length
      call attempt_integer_strict(NNN,the_words(which%line,3)%ch(1:NNN), &
                                order_lsq_cg_predictor    , trim(nf), which%line) 
    else if (the_words(iline,2)%ch(1:NNN) == 'ASPC') then
       cg_predict_restart_CTRL = 3
       if (cg_skip_Fourier%lskip) then
         N=5+2
       else
         N=5
       endif
       if (NumberOfWords(which%line) < N) then
        PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line, 'AT least one more record needed',&
                 ' always stable predictor corrector'
        STOP
       endif
       NNN=the_words(iline,3)%length
       call attempt_integer_strict(NNN,the_words(which%line,3)%ch(1:NNN), &
                                aspc_coars_Niters    , trim(nf), which%line)
       NNN=the_words(iline,4)%length
       call attempt_integer_strict(NNN,the_words(which%line,4)%ch(1:NNN), &
                                aspc_update_4full_iters    , trim(nf), which%line)
       NNN=the_words(iline,5)%length
       call attempt_real_strict(NNN,the_words(which%line,5)%ch(1:NNN), &
                                aspc_omega    , trim(nf), which%line)


    else
! AGAIN DEFAULT VALUES
    endif
print*, 'exit cg details cg_predict_restart_CTRL=',cg_predict_restart_CTRL
print*, 'exit cg details order_lsq_cg_predictor=',order_lsq_cg_predictor
 end subroutine CG_details

 subroutine get_rsmd
  use rsmd_data, only : rsmd
   integer NNN
   integer iline
   integer i,j,k,iv4(4),i_save
   real(8) f1
   call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'RSMD',l_skip_line,which,trim(nf),.false.)
   if (.not.which%find) then
      rsmd%any_request = .false.
      RETURN
   endif

      rsmd%any_request = .true.
      rsmd%print_details = .false.
      iline = which%line
      if (NumberOfWords(which%line) < 4) then
        PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line, 'when defining RSMD'
      STOP
      endif
       do k = 2, 4
       NNN = the_words(which%line,k)%length
       call attempt_integer_strict(NNN,the_words(which%line,k)%ch(1:NNN), &
                                   iv4(k) , trim(nf), which%line)
       if (iv4(k) < 1) then
         PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line,&
         'A INTEGER STRICT POSITIVE NUMBER REQUIRED',&
         'instead of the record:' ,iv4(k)
         STOP
       endif
       enddo
       rsmd%N_collect = iv4(2)
       rsmd%N_eval = iv4(3)
       rsmd%N_print = iv4(4)
       i_save = -1000
       do i = 2, NumberOfWords(which%line)
         NNN = the_words(which%line,i)%length
         if (the_words(which%line,i)%ch(1:NNN) == 'zb') then
              i_save = i
         endif
       enddo
       if (i_save < 0) then
          rsmd%N_Z_BINS = 100  ! default value
          print*, 'Default (100) value for rsmd%N_Z_BINS'
       else
          if (i_save == NumberOfWords(which%line)) then
             print*,'ERROR : ONE MORE INTEGER RECORD NEEDED in input file at line',which%line,&
             ' after the record "zb"'
             STOP
          endif
          NNN = the_words(which%line,i_save+1)%length
          call attempt_integer_strict(NNN,the_words(which%line,i_save+1)%ch(1:NNN), &
                                   rsmd%N_Z_BINS , trim(nf), which%line)
          if (rsmd%N_Z_BINS < 1) then
            PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line,&
            'A INTEGER STRICT POSITIVE NUMBER REQUIRED',&
            'instead of the record:' ,rsmd%N_Z_BINS
            STOP
          endif
        endif

       do i = 2, NumberOfWords(which%line)
         NNN = the_words(which%line,i)%length
         if (the_words(which%line,i)%ch(1:NNN) == '++') then
              rsmd%print_details = .true.
         endif
       enddo
       
       i_save = -1000
       do i = 2, NumberOfWords(which%line)
         NNN = the_words(which%line,i)%length
         if (the_words(which%line,i)%ch(1:NNN) == 'skip') then
             i_save = i 
         endif
       enddo
       if (i_save < 0) then
           f1 = 0.1d0
           print*,'DEFAULT value for rsmd%skip_times'
       else
          if (i_save == NumberOfWords(which%line)) then
             print*,'ERROR : ONE MORE REAL RECORD NEEDED in input file at line',which%line,&
             ' after the record "skip"'
             STOP
          endif
          NNN = the_words(which%line,i_save+1)%length
          call attempt_real_strict(NNN,the_words(which%line,i_save+1)%ch(1:NNN), &
                                   f1 , trim(nf), which%line)
          f1 = f1 / 100.0d0 
          if (f1 < 0.0d0) then
              print*,'ERROR : in input file at line',which%line,'a positiv real number is required instead of a negative one ',&
              the_words(which%line,i_save+1)%ch(1:NNN)
              STOP
          endif
        endif
        rsmd%skip_times = NINT(f1*dble(rsmd%N_print))
        if (rsmd%skip_times < 1) rsmd%skip_times=1
        if (rsmd%skip_times > rsmd%N_print) then 
          print*, 'TOO large value for rsmd%skip; it will be set to 0'
          rsmd%skip_times = 1
        endif

        if (rsmd%N_print < 3) then
               print*,'ERROR : in input file at line which%line',&
               ' the forth record must be an integer bigger than 3'
                STOP
        endif
        if (rsmd%N_print - rsmd%skip_times   < 2) then  
               print*,'ERROR : when definning rsmd%N_print and rsmd%skip_times; they do not satisfy',&
               'rsmd%N_print - rsmd%skip_times >=2 ; Increase rsmd%N_print or decrease rsmd%skip_times',&
               ': changes to be made in input file at line',which%line
               STOP
        endif
   ! if rsmd%print_details print 3D-rsmd  else print only diffusion coeficients
 end subroutine get_rsmd
!-------------------------

 subroutine get_rdfs
  use rdfs_data, only : rdfs,l_details_rdf_CTRL,N_BIN_rdf
  use array_math, only : order_vect
   integer NNN
   integer iline
   integer i,j,k,iv4(4),i_save
   real(8) f1
   call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'RDFS',l_skip_line,which,trim(nf),.false.)
   if (.not.which%find) then
      rdfs%any_request = .false.
      RETURN
   endif

      rdfs%any_request = .true.

      iline = which%line
      if (NumberOfWords(which%line) < 3) then
        PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line, 'when defining RSMD'
      STOP
      endif
       do k = 2, 3
       NNN = the_words(which%line,k)%length
       call attempt_integer_strict(NNN,the_words(which%line,k)%ch(1:NNN), &
                                   iv4(k) , trim(nf), which%line)
       if (iv4(k) < 1) then
         PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line,&
         'A INTEGER STRICT POSITIVE NUMBER REQUIRED',&
         'instead of the record:' ,iv4(k)
         STOP
       endif
       enddo
       rdfs%N_collect = iv4(2)
       rdfs%N_print = iv4(3)
  
i_save = -1000
       do i = 2, NumberOfWords(which%line)
         NNN = the_words(which%line,i)%length
         if (the_words(which%line,i)%ch(1:NNN) == 'r_b') then
              i_save = i
         endif
       enddo
       if (i_save < 0) then
          print*, 'Default (1) value for N_BINS_rdfs'
       else
          if (i_save == NumberOfWords(which%line)) then
             print*,'ERROR : ONE MORE INTEGER RECORD NEEDED in input file at line',which%line,&
             ' after the record "r_b"'
             STOP
           endif
          NNN = the_words(which%line,i_save+1)%length
          call attempt_integer_strict(NNN,the_words(which%line,i_save+1)%ch(1:NNN), &
                                  N_BIN_rdf , trim(nf), which%line)
          if (N_BIN_rdf < 1) then
            PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line,&
            'A INTEGER STRICT POSITIVE NUMBER REQUIRED',&
            'instead of the record:' ,N_BIN_rdf
            STOP
          endif
        endif


 
i_save = -1000
       do i = 2, NumberOfWords(which%line)
         NNN = the_words(which%line,i)%length
         if (the_words(which%line,i)%ch(1:NNN) == 'zb') then
              i_save = i
         endif
       enddo
       if (i_save < 0) then
          rdfs%N_Z_BINS = 1  ! default value
          print*, 'Default (1) value for rdfs%N_Z_BINS'
       else
          if (i_save == NumberOfWords(which%line)) then
             print*,'ERROR : ONE MORE INTEGER RECORD NEEDED in input file at line',which%line,&
             ' after the record "zb"'
             STOP
          endif
          NNN = the_words(which%line,i_save+1)%length
          call attempt_integer_strict(NNN,the_words(which%line,i_save+1)%ch(1:NNN), &
                                   rdfs%N_Z_BINS , trim(nf), which%line)
          if (rdfs%N_Z_BINS < 1) then
            PRINT*, 'ERROR : INSUFICIENT RECORDS in input file at line ',which%line,&
            'A INTEGER STRICT POSITIVE NUMBER REQUIRED',&
            'instead of the record:' ,rdfs%N_Z_BINS
            STOP
          endif
        endif

       do i = 2, NumberOfWords(which%line)
         NNN = the_words(which%line,i)%length
         if (the_words(which%line,i)%ch(1:NNN) == '++') then
              l_details_rdf_CTRL = .true.
         endif
       enddo

       
       i_save = -1000
       do i = 2, NumberOfWords(which%line)
         NNN = the_words(which%line,i)%length
         if (the_words(which%line,i)%ch(1:NNN) == 'pairs') then
             i_save = i
         endif
       enddo
       if (i_save < 0) then
           print*,'All ATOMIC PAIRS WILL BE TOKEN for drfs'
           rdfs%N_pairs = 1
           allocate(rdfs%what_input_pair(1:2)) ; rdfs%what_input_pair = -9999 ! for further reference
       else

          if (i_save + 1 > NumberOfWords(which%line)) then
             print*,'ERROR 1:  MORE RECORDS NEEDED in input file at line',which%line,&
             ' after the record "pairs"'
             print*, i_save+1, 'records=',NumberOfWords(which%line)
             STOP
          endif

          NNN = the_words(which%line,i_save+1)%length
          call attempt_integer_strict(NNN,the_words(which%line,i_save+1)%ch(1:NNN), &
                                   rdfs%N_pairs , trim(nf), which%line)
          
          if (rdfs%N_pairs < 1) then
             print*,'ERROR :  in input file at line',which%line,&
             ' Number of pairs cannot be zero or negative'
             STOP
          endif
          allocate(rdfs%what_input_pair(1:rdfs%N_pairs))

          if (i_save +1 + rdfs%N_pairs > NumberOfWords(which%line)) then
             print*,'ERROR 2:  MORE PAIRS NEEDED to be defined in input file at line',which%line,&
             ' after the record "pairs"'
             STOP
          endif
     
          do k = i_save +1 + 1 , i_save +1 + rdfs%N_pairs
             NNN = the_words(which%line,k)%length
             call attempt_integer_strict(NNN,the_words(which%line,k)%ch(1:NNN), &
                                   rdfs%what_input_pair(k-1-i_save) , trim(nf), which%line)
             if (rdfs%what_input_pair(k-1-i_save) < 1) then
              print*, 'ERROR in input file at line ',which%line, ' a positive integer required instead of ',&
                       rdfs%what_input_pair(k-1-i_save)
              STOP
             endif
          enddo           

       endif

       do i = 1, rdfs%N_pairs
       if(rdfs%what_input_pair(i) > 0)then
       do j = i+1,rdfs%N_pairs
       if(rdfs%what_input_pair(j) > 0)then  ! negative case is when they are not declared here, and will be token all
         if (rdfs%what_input_pair(i)==rdfs%what_input_pair(j)) then
             print*, 'ERROR; redundant atomic pairs in input file at line: ',which%line, &
             'The pair',rdfs%what_input_pair(i),'is declared twice '
             print*,' Delete one of the records at colomns ', i_save + i + 1 ,i_save + j + 1, ' line ',which%line
             STOP
         endif
       endif
       enddo
       endif
       enddo

    if (mod(rdfs%N_print,rdfs%N_collect) /= 0) then
      print*, 'ERROR in input file at line',which%line,' first 2 nubmer must divide by each other',&
      '(mod(rdfs%N_print rdfs%N_collect) must be zero and it is not; '
      STOP 
    endif
    call order_vect(rdfs%what_input_pair)

!print*, 'exiting rdfs'
!print*,'details=',l_details_rdf_CTRL
!print*, 'rdfs=',rdfs%N_print,rdfs%N_Z_bins,rdfs%N_collect,rdfs%N_pairs
!print*, 'rdfs%what_input_pair=',rdfs%what_input_pair
 end subroutine get_rdfs
!-------------------------
 subroutine get_type_ensamble(i_type_ensamble,i_type_thermos,i_type_barr)
 integer, intent(OUT) :: i_type_ensamble,i_type_thermos,i_type_barr
 integer NNN,iline,kkk
 logical l_error
 call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'ENSAMBLE',l_skip_line,which,trim(nf),.false.)
 if (.not.which%find) then
        print*, 'WARNING : NO ENSAMBLE DEFINED; THE PROGRAM WILL DO NVE'
        i_type_ensamble = 0 !
        i_type_thermos=0
        i_type_barr=0
        RETURN
 endif
 if (NumberOfWords(which%line) < 2) then
       print*, 'WARNING : NO ENSAMBLE DEFINED; THE PROGRAM WILL DO NVE'
       i_type_ensamble = 0
       i_type_thermos=0
       i_type_barr=0
       RETURN
 endif


 iline = which%line
 NNN = the_words(iline,2)%length

 call ensamble_def(the_words(iline,2)%ch(1:NNN),i_type_ensamble,i_type_thermos,i_type_barr,l_error   )
 thermo_coupling=0.0d0 ; barostat_coupling=0.0d0 ;
 if (l_error) then
   print*, 'ERROR in the file "',trim(nf),'"at line: ',which%line,' when defining the ENSAMBLE'
   print*, ' The keyword "',the_words(iline,2)%ch(1:NNN),'" do not pre-define any ensamble'
   STOP
 end if

 if (i_type_thermos > 0) then
   call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'THERMOSTAT_COUPLING',l_skip_line,which,trim(nf),.false.)
   if (.not.which%find) then
        print*, 'ERROR in "',trim(nf),'"file : NO THERMOSTAT_COUPLING DEFINED; THE PROGRAM WILL STOP'
        STOP
    endif
    if (NumberOfWords(which%line) < 2) then
       print*, 'ERROR in "',trim(nf),'"file : NO THERMOSTAT_COUPLING DEFINED; THE PROGRAM WILL STOP'
       STOP
    endif
    NNN = the_words(which%line,2)%length
    call attempt_real_strict(NNN,the_words(which%line,2)%ch(1:NNN), &
                                   thermo_coupling , trim(nf), which%line)
  endif

  if (i_type_barr > 0) then
     call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'BAROSTAT_COUPLING',l_skip_line,which,trim(nf),.false.)
     if (.not.which%find) then
        print*, 'ERROR in "',trim(nf),'"file : NO THERMOSTAT_COUPLING DEFINED; THE PROGRAM WILL STOP'
        STOP
    endif
    if (NumberOfWords(which%line) < 4) then
       print*, 'ERROR in "',trim(nf),'"file. At line',which%line,'4 records are required '
       STOP
    endif
    do i = 1, 3
      kkk = 1+i
      NNN = the_words(which%line,kkk)%length
      call attempt_real_strict(NNN,the_words(which%line,kkk)%ch(1:NNN), &
                                   barostat_coupling(i) , trim(nf), which%line)
    enddo
   endif ! i_type_barr > 0

!    get temperature
if (i_type_thermos > 0) then
    call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'TEMPERATURE',l_skip_line,which,trim(nf),.false.)
    if (.not.which%find) then
        print*, 'ERROR in "',trim(nf),'"file : THE TEMPERATURE IS NOT DEFINED; THE PROGRAM WILL STOP'
        STOP
    endif
    if (NumberOfWords(which%line) < 2) then
       print*, 'ERROR in "',trim(nf),'"file :  THE TEMPERATURE IS NOT SPECIFIED ; THE PROGRAM WILL STOP'
       STOP
    endif
    NNN = the_words(which%line,2)%length
    call attempt_real_strict(NNN,the_words(which%line,2)%ch(1:NNN), &
                                   temperature , trim(nf), which%line)
endif
if (i_type_thermos == 0) then   ! NVE case - still I assign an input T
    call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'TEMPERATURE',l_skip_line,which,trim(nf),.false.)
    if (.not.which%find) then
        print*, 'In "',trim(nf),'"file : THE TEMPERATURE IS NOT DEFINED; IT will be assigned at 50K'
        temperature=50.0d0
    else
    if (NumberOfWords(which%line) < 2) then
       print*, 'ERROR in "',trim(nf),'"file :  THE TEMPERATURE IS NOT SPECIFIED ; THE PROGRAM WILL STOP'
       STOP
    endif
    NNN = the_words(which%line,2)%length
    call attempt_real_strict(NNN,the_words(which%line,2)%ch(1:NNN), &
                                   temperature , trim(nf), which%line)
    endif
endif

!    get pressure

if (i_type_barr > 0) then
   call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'PRESSURE',l_skip_line,which,trim(nf),.false.)
   if (.not.which%find) then
        print*, 'ERROR in "',trim(nf),'"file : THE PRESSURE IS NOT DEFINED; THE PROGRAM WILL STOP'
        STOP
    endif
    if (NumberOfWords(which%line) < 4) then
       print*, 'ERROR in "',trim(nf),'"file :  4 records needed when specify the pressure at line ',&
       which%line,' you have fewer than that; THE PROGRAM WILL STOP'
       STOP
    endif
    NNN = the_words(which%line,2)%length
    call attempt_real_strict(NNN,the_words(which%line,2)%ch(1:NNN), &
                                   pressure_xx , trim(nf), which%line)
    NNN = the_words(which%line,3)%length
    call attempt_real_strict(NNN,the_words(which%line,3)%ch(1:NNN), &
                                   pressure_yy , trim(nf), which%line)
    NNN = the_words(which%line,4)%length
    call attempt_real_strict(NNN,the_words(which%line,4)%ch(1:NNN), &
                                   pressure_zz , trim(nf), which%line)
endif ! (i_type_barr > 0) then



 end subroutine get_type_ensamble

 subroutine get_excluded_14_flags
 
 use chars, only : UP_CASE, locate_UPCASE_word_in_key
 use connectivity_ALL_data, only : red_14_vdw,red_14_Q, red_14_Q_mu, &
                  l_exclude_14_dih_CTRL,l_exclude_14_all_CTRL, &
                  l_red_14_vdw_CTRL, l_red_14_Q_CTRL, l_red_14_Q_mu_CTRL, &
                  l_build_14_from_dih_CTRL, l_build_14_from_angle_CTRL,&
                  l_red_14_mu_mu_CTRL,red_14_mu_mu

 implicit none
 integer NNN,iline,kkk
 logical l_error
 integer i_position
 logical l_found,l_1

 call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'EXCLUDE_14',l_skip_line,which,trim(nf),.false.)
 kkk = which%line

 call locate_UPCASE_word_in_key('DIH', 1, the_words(kkk,:),l_found,i_position)
 if (l_found) then
    i_position=i_position+1
    NNN = the_words(kkk,i_position)%length
    l_1= NNN==3.and.UP_CASE(the_words(kkk,i_position)%ch(1:1))=='Y'
    l_1=l_1.and.UP_CASE(the_words(kkk,i_position)%ch(2:2))=='E'
    l_1=l_1.and.UP_CASE(the_words(kkk,i_position)%ch(3:3))=='S' 
    l_build_14_from_dih_CTRL=l_1
 endif

 
 call  locate_UPCASE_word_in_key('ALL', 1, the_words(kkk,:),l_found,i_position)
 if (l_found) then
    i_position=i_position+1
    NNN = the_words(kkk,i_position)%length
    l_1= (NNN==3.and.UP_CASE(the_words(kkk,i_position)%ch(1:1))=='Y') 
    l_1=l_1.and.UP_CASE(the_words(kkk,i_position)%ch(2:2))=='E'
    l_1=l_1.and.UP_CASE(the_words(kkk,i_position)%ch(3:3))=='S'  
    l_build_14_from_angle_CTRL=l_1
 endif

print*, 'l_exclude_14_dih_CTRL l_exclude_14_all_CTRL=',l_exclude_14_dih_CTRL,l_exclude_14_all_CTRL

 if (.not.(l_build_14_from_dih_CTRL.or.l_build_14_from_angle_CTRL)) then
    print*, 'ERROR when biuld 14-info in get_excluded_14_flags',&
            ' l_build_14_from_dih_CTRL.and.l_build_14_from_angle_CTRL are both false; one of them must be true',&
            ' Modify in input file at line ',kkk, ' lll=',l_build_14_from_dih_CTRL,l_build_14_from_angle_CTRL
    STOP
 endif

 call locate_UPCASE_word_in_key('VDW', 1, the_words(kkk,:),l_found,i_position)
 if (l_found) then
    i_position=i_position+1
    NNN = the_words(kkk,i_position)%length
    call attempt_real_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                                  red_14_vdw, trim(nf), kkk)
 else
    print*, 'DEFALUT VALUE OF 14-reduced vdw will be used: red_14_vdw=',red_14_vdw
 endif

  call locate_UPCASE_word_in_key('Q-Q', 1, the_words(kkk,:),l_found,i_position)
 if (l_found) then
    i_position=i_position+1
    NNN = the_words(kkk,i_position)%length
    call attempt_real_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                                  red_14_Q, trim(nf), kkk)
 else
    print*, 'DEFALUT VALUE OF 14-reduced Q will be used: red_14_Q=',red_14_Q
 endif

  call locate_UPCASE_word_in_key('Q-MU', 1, the_words(kkk,:),l_found,i_position)
 if (l_found) then
    i_position=i_position+1
    NNN = the_words(kkk,i_position)%length
    call attempt_real_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                                  red_14_Q_mu, trim(nf), kkk)
 else
    print*, 'DEFALUT VALUE OF 14-reduced Q-mu will be used: red_14_Q_mu=',red_14_Q_mu
 endif

 call locate_UPCASE_word_in_key('MU-MU', 1, the_words(kkk,:),l_found,i_position)
  if (l_found) then
    i_position=i_position+1
    NNN = the_words(kkk,i_position)%length
    call attempt_real_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                                  red_14_mu_mu, trim(nf), kkk)
 else
    print*, 'DEFALUT VALUE OF 14-reduced Q-mu will be used: red_14_mu_mu=',red_14_mu_mu
 endif


 l_red_14_vdw_CTRL = dabs(red_14_vdw )> 1.0d-3
 l_red_14_Q_CTRL   = dabs(red_14_Q   )> 1.0d-3
 l_red_14_Q_mu_CTRL= dabs(red_14_Q_mu)> 1.0d-3
 l_red_14_mu_mu_CTRL= dabs(red_14_mu_mu)> 1.0d-3

print*, 'red_14_vdw Q Q-mu=',red_14_vdw,red_14_Q,red_14_Q_mu, &
l_red_14_vdw_CTRL,l_red_14_Q_CTRL,l_red_14_Q_mu_CTRL
 end subroutine get_excluded_14_flags
 

 subroutine get_profiles
 integer NNN,kkk


    call  search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'PROFILES',l_skip_line,which,trim(nf),.false.)  
    if (.not.which%find) then
      l_1st_profile_CTRL = .false. ; l_2nd_profile_CTRL=.false.
      l_need_2nd_profile=.false. ; l_need_1st_profile=.false.
print*, 'NO PROFILING REQUEST'
      RETURN
    endif 
    if (NumberOfWords(which%line) < 3) then
       print*, 'ERROR in "',trim(nf),'"file. At line',which%line,'more records are required '
       STOP
    endif
    kkk = 2 
    NNN = the_words(which%line,kkk)%length
    if (the_words(which%line,kkk)%ch(1:NNN) == 'NO') then
      l_1st_profile_CTRL = .false. ; l_need_1st_profile=.false. 
    elseif (the_words(which%line,kkk)%ch(1:NNN) == 'YES') then
      l_1st_profile_CTRL=.true.; 
    else
      print*, 'ERROR in "',trim(nf),'"file. At line',which%line,'have YES or NO '
       STOP
    endif
    kkk = 3
    NNN = the_words(which%line,kkk)%length
    if (the_words(which%line,kkk)%ch(1:NNN) == 'NO') then
      l_2nd_profile_CTRL = .false. ; l_need_2nd_profile=.false.
    elseif (the_words(which%line,kkk)%ch(1:NNN) == 'YES') then
      l_2nd_profile_CTRL=.true.;
    else
      print*, 'ERROR in "',trim(nf),'"file. At line',which%line,'have YES or NO '
       STOP
    endif
    if (l_1st_profile_CTRL.or.l_2nd_profile_CTRL) then
      if (NumberOfWords(which%line) < 6) then
        print*, ' Z ERROR in "',trim(nf),'"file. At line',which%line,'more records are required '
       STOP 
      endif       
    endif
if ((.not.l_1st_profile_CTRL).and.(.not.l_2nd_profile_CTRL) ) RETURN

     
    kkk=4 
    NNN = the_words(which%line,kkk)%length
    if (the_words(which%line,kkk)%ch(1:NNN) == 'z'.or.the_words(which%line,kkk)%ch(1:NNN)=='Z') then
    kkk = 5
    NNN = the_words(which%line,kkk)%length
      call attempt_integer_strict(NNN,the_words(which%line,kkk)%ch(1:NNN), &
                                  N_BINS_ZZ  , trim(nf), which%line) 
      N_BINS_XX=N_BINS_ZZ
      N_BINS_YY=N_BINS_ZZ ! may need to change that
    kkk = 6
    NNN = the_words(which%line,kkk)%length
      call attempt_integer_strict(NNN,the_words(which%line,kkk)%ch(1:NNN), &
                                  N_BINS_ZZs  , trim(nf), which%line)

    elseif (the_words(which%line,kkk)%ch(1:NNN) == 'XY'.or.the_words(which%line,kkk)%ch(1:NNN)=='xy')then
      if (NumberOfWords(which%line) < 7) then
        print*, ' XY ERROR in "',trim(nf),'"file. At line',which%line,'more records are required '
       STOP
      endif
      kkk=5
      NNN = the_words(which%line,kkk)%length
      call attempt_integer_strict(NNN,the_words(which%line,kkk)%ch(1:NNN), &
                                  N_BINS_XX  , trim(nf), which%line)
      kkk=6
      NNN = the_words(which%line,kkk)%length
      call attempt_integer_strict(NNN,the_words(which%line,kkk)%ch(1:NNN), &
                                  N_BINS_YY  , trim(nf), which%line)

    elseif (the_words(which%line,kkk)%ch(1:NNN) == 'XYZ'.or.the_words(which%line,kkk)%ch(1:NNN)=='xyz')then
      if (NumberOfWords(which%line) < 8) then
        print*, 'XYZ ERROR in "',trim(nf),'"file. At line',which%line,'more records are required '
       STOP
      endif
      kkk=5
      NNN = the_words(which%line,kkk)%length
      call attempt_integer_strict(NNN,the_words(which%line,kkk)%ch(1:NNN), &
                                  N_BINS_XX  , trim(nf), which%line)
      kkk=6
      NNN = the_words(which%line,kkk)%length
      call attempt_integer_strict(NNN,the_words(which%line,kkk)%ch(1:NNN), &
                                  N_BINS_YY  , trim(nf), which%line)
      kkk=7

      NNN = the_words(which%line,kkk)%length
      call attempt_integer_strict(NNN,the_words(which%line,kkk)%ch(1:NNN), &
                                  N_BINS_ZZ  , trim(nf), which%line)
 
    else
     print*, '\end ERROR in "',trim(nf),'"file. At line',which%line,'Syntax ERROR'
     STOP
    endif
 end subroutine get_profiles 


   subroutine get_system_preparation
    use sys_preparation_data
    use extra_line_module
    implicit none
    logical l_found, l1,l2,l3
    integer i,j,k,i_position,NNN,kkk,ijk
 
    call sys_prep_default
    call search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'SYSTEM_PREPARATION',l_skip_line,which,trim(nf),.false.)

    print*, 'sys prep found?',which%find
    if (.not.which%find) RETURN ! This is a job for good!
    sys_prep%where_in_file = which%line
    kkk = which%line
    call locate_UPCASE_word_in_key('ADJUST_BOX_TO', 3, the_words(kkk,:),l_found,i_position)
    l1=l_found
    if (l_found) then
    sys_prep%type_prep = 0 ! adjust boxes
    do ijk=1,3
      i_position=i_position+1
       NNN = the_words(kkk,i_position)%length
       call attempt_real_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                           sys_prep%box_to(ijk)  , trim(nf), kkk)
    enddo!ijk
    endif
    call locate_UPCASE_word_in_key('ADJUST_SFC_BY',1, the_words(kkk,:),l_found,i_position)
    l2=l_found
    if (l1.and.l2) then
     print*, 'ERROR in input file ',trim(nf),' at line ',sys_prep%where_in_file, &
     ' BOTH records SYSTEM_PREPARATION and ADJUST_BOX_TO are present ',&
     ' It may be better to adjust one thing at the time so comment one of those ',&
     ' records at line', sys_prep%where_in_file, 'and restart the code. The program will now stop'
     STOP
    endif
    if (l_found) then
    sys_prep%type_prep = 1 ! adjust sfc atoms
      i_position=i_position+1
       NNN = the_words(kkk,i_position)%length
       call attempt_real_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                           sys_prep%zsfc_by  , trim(nf), kkk)
    endif
    call locate_UPCASE_word_in_key('ADD_Z_TO_MOLTYPE',1, the_words(kkk,:),l_found,i_position)
    l3 = l_found 
    if ((l1.and.l3).or.(l2.and.l3)) then
     print*, 'ERROR in input file ',trim(nf),' at line ',sys_prep%where_in_file, &
     ' BOTH records SYSTEM_PREPARATION and ADJUST_BOX_TO are present ',&
     ' It may be better to adjust one thing at the time so comment one of those ',&
     ' records at line', sys_prep%where_in_file, 'and restart the code. The program will now stop'
       STOP
    endif
    if (l3) then
      allocate(extra_line(NumberOfWords(kkk)-i_position))
      do ijk  = i_position + 1, NumberOfWords(kkk)
print*,ijk,NumberOfWords(kkk),trim(the_words(kkk,ijk)%ch)
      extra_line(ijk-i_position)%ch = the_words(kkk,ijk)%ch
      extra_line(ijk-i_position)%length = the_words(kkk,ijk)%length
      enddo
      extra_line_length = NumberOfWords(kkk) - i_position
      sys_prep%type_prep = 2  ! By MolType
      extra_line_line=kkk
      sys_prep%where_in_file = which%line
    endif
    sys_prep%any_prep=l1.or.l2.or.l3 

    if (sys_prep%any_prep) then
      print*, 'WARNING : THIS JOB IS A SYSTEM PREPARATION JOB ONLY!!!!'
      if (l1) then
      print*, 'THE BOXES XX YY ZZ WILL BE ABJUSTED TO A FINAL VALUE OF : ',sys_prep%box_to(:)
      endif
      if (l2) then
      print*, 'THE ZZ position of sfc constrained atoms (like electrodes) ',&
              'WILL BE ABJUSTED BY AN INCREMENT OF : ',sys_prep%zsfc_by
      endif
      print*, 'WARNING : THIS JOB IS A SYSTEM PREPARATION JOB ONLY!!!!'
      print*, 'WARNING : THIS JOB IS A SYSTEM PREPARATION JOB ONLY!!!!'
      print*, 'IF you wish to run a production run then just comment the record',&
      ' SYSTEM_PREPARATION  from the line ',sys_prep%where_in_file, 'in the file ',trim(nf)
      print*,'----------------------------------------------------'
    endif 
    
    print*, 'sys_prep%any_prep=',sys_prep%any_prep
 
   end subroutine get_system_preparation
   
   subroutine get_shake_tol
   use shake_data
    integer kkk,NNN
        call search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'SHAKE_TOLERANCE',l_skip_line,which,trim(nf),.false.)
        if (which%find) then
        if (NumberOfWords(which%line)>1) then
           kkk = 2
           NNN = the_words(which%line,kkk)%length
           call attempt_real_strict(NNN,the_words(which%line,kkk)%ch(1:NNN), &
                        SHAKE_TOLERANCE  , trim(nf), kkk)
print*,'SHAKE_TOLERANCE redefined as: ',SHAKE_TOLERANCE
        else
           print*,'WARNING: More records needed when define the shake tolerance; ',&
           'The reading from file ',trim(nf),' and line ', which%line, 'will be ignored; ',&
           'The default value of SHAKE_TOLERANCE=',&
           SHAKE_TOLERANCE, ' will be used'
        endif
        else
        print*, 'A default SHAKE_TOLERANCE=',SHAKE_TOLERANCE, ' will be used (if necesary)'
        endif
   end subroutine get_shake_tol

   subroutine get_quick_stat_preview
     use quick_preview_stats_data
     integer kkk,NNN,i_position
     logical l_found
     call quick_preview_stats_default
     call search_words(1,lines,lines,Max_words_per_line,the_words,&
                 SizeOfLine,NumberOfWords,'QUICK_PREVIEW',l_skip_line,which,trim(nf),.false.)
     if (which%find) then
     kkk = which%line
     call locate_UPCASE_word_in_key('MORE_ENERGIES',1, the_words(which%line,:),l_found,i_position)      
     l_print_more_energies_CTRL = l_found
     if (l_found) then
       i_position=i_position+1
       NNN = the_words(kkk,i_position)%length
       call attempt_integer_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                           N_PRINT_MORE_ENERGIES  , trim(nf), kkk)
       if (N_PRINT_MORE_ENERGIES < 1) l_print_more_energies_CTRL = .false.
      endif
      call locate_UPCASE_word_in_key('MORE_STATS',1, the_words(kkk,:),l_found,i_position)
      quick_preview_stats%any_request = l_found
      if (l_found) then
        i_position=i_position+1
        NNN = the_words(kkk,i_position)%length
        call attempt_integer_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                           quick_preview_stats%how_often  , trim(nf), kkk)
        if (quick_preview_stats%how_often<1) quick_preview_stats%any_request = .false.
      endif 
      endif ! which%find

print*,'l_print_more_energies_CTRL=',l_print_more_energies_CTRL
print*,'N_PRINT_MORE_ENERGIES=',N_PRINT_MORE_ENERGIES
print*,'quick_preview_stats=',quick_preview_stats
   end subroutine get_quick_stat_preview

  subroutine get_temperature_annealing
  use temperature_anneal_data
  implicit none
  integer kkk,NNN,i_position
  logical l_found

  call defauts_temperature_anneal_data

  call search_words(1,lines,lines,Max_words_per_line,the_words,&
                SizeOfLine,NumberOfWords,'TEMPERATURE_ANNEALING',l_skip_line,which,trim(nf),.false.)
   anneal_T%any_Tanneal = which%find
   if (.not.which%find) RETURN
   kkk = which%line
   if (NumberOfWords(kkk) < 3 ) then
    print*, 'ERROR more records needed at file ',trim(nf),' line ',kkk
    STOP
   endif 
   NNN = the_words(kkk,2)%length 
   call attempt_real_strict(NNN,the_words(kkk,2)%ch(1:NNN), &
                           anneal_T%Tstart  , trim(nf), kkk)
   NNN = the_words(kkk,3)%length
   call attempt_real_strict(NNN,the_words(kkk,3)%ch(1:NNN), &
                           anneal_T%Tend  , trim(nf), kkk)
   temperature = anneal_T%Tstart
   anneal_T%dT = anneal_T%Tend - anneal_T%Tstart
  end subroutine get_temperature_annealing

  subroutine get_put_CM_molecules_in_box
   use CTRLs_data, only : put_CM_molecules_in_box_CTRL
     integer kkk,NNN,i_position
     logical l_found
     put_CM_molecules_in_box_CTRL=.true.
     call search_words(1,lines,lines,Max_words_per_line,the_words,&
                SizeOfLine,NumberOfWords,'PUT_CM_MOLECULES_IN_BOX',l_skip_line,which,trim(nf),.false.)
     if (which%find) then 
       call locate_UPCASE_word_in_key('NO',0, the_words(which%line,:),l_found,i_position)
       if (l_found.and.(i_position==2)) then
          put_CM_molecules_in_box_CTRL=.false.
          print*,'WARNING: THE molecules were not recentered in box '
       else
       put_CM_molecules_in_box_CTRL = .true.
       print*, 'PARSER MESSAGE: put_CM_molecules_in_box_CTRL = true ! The position of molecular CM will be recentered in box'
       endif
     endif 
  end subroutine get_put_CM_molecules_in_box

   subroutine get_history
   use history_data
     integer kkk,NNN,i_position
     logical l_found

    call default_history
     call search_words(1,lines,lines,Max_words_per_line,the_words,&
                SizeOfLine,NumberOfWords,'HISTORY',l_skip_line,which,trim(nf),.false.)
     history%any_request=which%find
     if (which%find) then
       kkk = which%line
       call locate_UPCASE_word_in_key('HOW_OFTEN',1, the_words(which%line,:),l_found,i_position)
       l_print_more_energies_CTRL = l_found
       if (.not.l_found) then
       print*,'ERROR in file ',trim(nf), ' at line ',kkk,' when request recording the history:',&
       ' The record how_often is not specified'
       STOP
       endif
       i_position=i_position+1
       NNN = the_words(kkk,i_position)%length
       call attempt_integer_strict(NNN,the_words(kkk,i_position)%ch(1:NNN), &
                           history%how_often  , trim(nf), kkk)
       if (history%how_often<0) history%any_request=.false.
       call locate_UPCASE_word_in_key('CEL',0, the_words(which%line,:),l_found,i_position)
       if (l_found) then
         history%cel=1
       else
         history%cel=0
       endif
       call locate_UPCASE_word_in_key('X',0, the_words(which%line,:),l_found,i_position)
       if (l_found) then
         history%x=1
       else
         history%x=0
       endif
       call locate_UPCASE_word_in_key('V',0, the_words(which%line,:),l_found,i_position)
       if (l_found) then
         history%v=1
       else
         history%v=0
       endif
       call locate_UPCASE_word_in_key('F',0, the_words(which%line,:),l_found,i_position)
       if (l_found) then
         history%f=1
       else
         history%f=0
       endif
      call locate_UPCASE_word_in_key('EN',0, the_words(which%line,:),l_found,i_position)
       if (l_found) then
         history%en=1
       else
         history%en=0
       endif

     endif
  if (history%any_request) then
    if(.not.(history%cel.or.history%x.or.history%v.or.history%f)) then
      print*,'WARNING: NO DATA is will be written in HISTORY file except for the header'
    endif
  endif

 print*,'history=',history
 
   end subroutine get_history




subroutine parse_extra_line
use sys_preparation_data
use extra_line_module
use mol_type_data, only : N_type_molecules
implicit none
logical is_OK,first_time
integer i,j,k,i1,icount,kstart,istart,kend,iend,k1,k2,ki,kf
integer, parameter :: depth = 2
integer, allocatable :: where_k_start(:),where_k_end(:),where_i_start(:),where_i_end(:),elements(:)
integer, allocatable :: which_elements(:,:)
real(8),allocatable :: params(:)
character(250) mychar

first_time=.true.
is_OK=.false.
i1 = 0
icount=0
if (sys_prep%type_prep == 2) then
do i = 1, extra_line_length
print*,i,extra_line_length,trim( extra_line(i)%ch)
 do k = 1, extra_line(i)%length
  if (extra_line(i)%ch(k:k)=='(') then
    i1 = i1 + 1
    if (i1 == 1.and.first_time) then
       kstart=k
       istart=i
       first_time=.false.
    endif
    if (i1==depth) is_OK=.true.
    if (i1 > depth) then
      print*,'SYNTAX ERROR in in.in at line',extra_line_line, ' too many ()'
      STOP
    endif
  endif
  if (extra_line(i)%ch(k:k)==')') then
    i1 = i1 - 1
    icount=icount + 1
    if (i1==0) then
       kend = k
       iend = i
       goto 4
    endif
   endif
  enddo
enddo

4 continue
icount=icount-1;
print*,'i1=',i1

if (.not.is_OK)then
print*, 'ERROR in in.in file. SYNTAX ERROR. The Mol Types must be between paranthesis. see line',extra_line_line
stop
endif
if (first_time) then
print*, 'ERROR in in.in file. At least one pharantesis must be opened in in.in file at line',extra_line_line
stop
endif
if (i1 > 0) then
print*, 'ERROR in in.in file. Too many paranthesis ( opened or too few ) closed in file in.in at line',extra_line_line
stop
else if (i1 < 0) then
print*, 'ERROR in in.in file. Too many paranthesis ) closed ot too many ( oppened in file in.in at line',extra_line_line
stop
endif

N_actions_sys_prep = icount
print*,'N_actions_sys_prep=',N_actions_sys_prep
allocate(where_k_start(N_actions_sys_prep),where_k_end(N_actions_sys_prep),&
         where_i_start(N_actions_sys_prep),where_i_end(N_actions_sys_prep))



i1=0
do i = istart, iend
 if (i==istart) then
    k1 = kstart
 else
    k1 = 1
 endif
 if (i==iend) then
    k2 = kend
 else
    k2 = extra_line(i)%length
 endif
 do k = k1,k2
  if (extra_line(i)%ch(k:k)=='(') then
    i1 = i1 + 1
    if (i1 /=1 ) then
      where_k_start(i1-1)=k
      where_i_start(i1-1)=i
    endif
  endif
 enddo
enddo

i1=0
do i = istart,iend
 if (i==istart) then
    k1 = kstart
 else
    k1 = 1
 endif
 if (i==iend) then
    k2 = kend
 else
    k2 = extra_line(i)%length
 endif
 do k = 1, extra_line(i)%length
  if (extra_line(i)%ch(k:k)==')') then
    i1 = i1 + 1
    if (i1<=N_actions_sys_prep) then
      where_k_end(i1)=k
      where_i_end(i1)=i
    endif
   endif
 enddo
enddo

print*,'where_k_start=',where_k_start
print*,'where_k_end=',where_k_end
print*,'where_i_start=',where_i_start
print*,'where_i_end=',where_i_end

do i = 1, N_actions_sys_prep
if (where_k_start(i)<1) then
 print*,'ERROR in parse_extra_line parser (reading in.in line',extra_line_line,') (where_k_start(i)<1)'
 stop
endif
if (where_k_end(i)<1) then
 print*,'ERROR in parse_extra_line parser (reading in.in line',extra_line_line,') (where_k_end(i)<1)'
 stop
endif
if (where_i_start(i)<1) then
 print*,'ERROR in parse_extra_line parser (reading in.in line',extra_line_line,') (where_i_start(i)<1)'
 stop
endif
if (where_i_end(i)<1) then
 print*,'ERROR in parse_extra_line parser (reading in.in line',extra_line_line,') (where_i_end(i)<1)'
 stop
endif
if (where_i_start(i) >where_i_end(i))then
 print*,'ERROR in parse_extra_line parser reading in.in' 
 print*,'The paranthesis are in not in right order in in.in file at line',extra_line_line
 stop
endif
if (where_i_start(i)==where_i_end(i).and.where_k_start(i)>where_i_end(i))then
 print*,'ERROR in parse_extra_line parser reading in.in line'
 print*,'The paranthesis are in not in right order in in.in file at line',extra_line_line
 stop
endif
enddo

! Validation complete ; now parse
allocate(elements(N_actions_sys_prep),params(N_actions_sys_prep));elements=0
do i = 1, N_actions_sys_prep
  j=extra_line(where_i_start(i))%length 
  if (j>1.and. where_k_start(i)<j ) elements(i)=elements(i)+1 
  do k1=where_i_start(i)+1,where_i_end(i)-1
       elements(i)=elements(i)+1
  enddo
  j=extra_line(where_i_end(i))%length
  if (j>1.and. where_k_end(i)>1.and.where_i_start(i)/=where_i_end(i) ) elements(i)=elements(i)+1
  if (where_i_start(i)==where_i_end(i) .and.where_k_start(i)==where_i_end(i)-1) then
    print*,'ERROR in parse_extra_line parser reading in.in at line' ,extra_line_line
    print*,' DELETE the empty sequence ()'
    STOP
  endif
enddo


!print*,'elements=',elements

do i = 1, N_actions_sys_prep
 j=where_i_end(i)+1
 NNN=extra_line(j).length
 call attempt_real_strict(NNN, extra_line(j)%ch(1:NNN) , params(i), 'in.in',extra_line_line)
enddo
!print*,'params=',params

allocate(which_elements(N_actions_sys_prep,maxval(elements)))
elements=0;

do i = 1, N_actions_sys_prep
  j=extra_line(where_i_start(i))%length
  if (j>1.and. where_k_start(i)<j ) then 
     elements(i)=elements(i)+1
     NNN=j-where_k_start(i)      
     mychar(1:NNN) = extra_line(where_i_start(i))%ch(where_k_start(i)+1:j)
     call attempt_integer_strict(NNN, mychar(1:NNN) , which_elements(i,elements(i)), 'in.in',extra_line_line)
!print*,i,'which_element=',which_elements(i,elements(i))
  endif
  do k1=where_i_start(i)+1,where_i_end(i)-1
       elements(i)=elements(i)+1
       NNN=extra_line(k1)%length
       mychar(1:NNN) = extra_line(k1)%ch(1:NNN)
       call attempt_integer_strict(NNN, mychar(1:NNN) , which_elements(i,elements(i)), 'in.in',extra_line_line)
!print*,i,'*which_element=',which_elements(i,elements(i))
  enddo
  j=extra_line(where_i_end(i))%length
  if (j>1.and. where_k_end(i)>1.and.where_i_start(i)/=where_i_end(i) ) then 
     elements(i)=elements(i)+1
     NNN=where_k_end(i)-1
     mychar(1:NNN) = extra_line(where_i_end(i))%ch(1:where_k_end(i)-1)
     call attempt_integer_strict(NNN, mychar(1:NNN) , which_elements(i,elements(i)), 'in.in',extra_line_line)
!print*,i,'**which_element=',which_elements(i,elements(i))
  endif
  if (where_i_start(i)==where_i_end(i) .and.where_k_start(i)==where_i_end(i)-1) then
    print*,'ERROR in parse_extra_line parser reading in.in at line' ,extra_line_line
    print*,' DELETE the empty sequence ()'
    STOP
  endif
enddo


do i = 1, N_actions_sys_prep
do k1=1,elements(i)
  if (which_elements(i,k1) > N_type_Molecules) then
     print*,'ERROR in in.in file at line ',extra_line_line
     print*,'which_elements out of range > N_type_Molecules'
     stop
  endif
  if (which_elements(i,k1) < 1) then
     print*,'ERROR in in.in file at line ',extra_line_line
     print*,'which_elements out of range (<=0) !SEVERE!!!!! '
     stop
  endif

do k2=k1+1,elements(i)
if (k1/=k2)then
  if (which_elements(i,k1)==which_elements(i,k2)) then
print*,'ERROR in in.in file at line ',extra_line_line
print*,'a molecule cannot be acted upon twice : remove duplicates '
stop
  endif
endif
enddo ! k2
do j = i +1, N_actions_sys_prep
if(i/=j) then
 do k2=1,elements(j)
   if (which_elements(i,k1)==which_elements(j,k2)) then
      print*,'ERROR in in.in file at line ',extra_line_line
      print*,'a molecule cannot be acted upon in two different actions '
      print*,'remove duplicate molecules at line ',extra_line_line
      stop
   endif
 enddo
endif
enddo
enddo
enddo
allocate(move_Zmol_type_sys_prep(N_type_molecules));move_Zmol_type_sys_prep(:) = 0.0d0;
do i = 1, N_actions_sys_prep
do k1=1,elements(i)
   j=which_elements(i,k1)
   move_Zmol_type_sys_prep(j) = params(i)
enddo  
enddo

deallocate(where_k_start,where_k_end,where_i_start,where_i_end)
deallocate(extra_line)
deallocate(which_elements,elements,params)

print*,'move_Zmol_type_sys_prep=',move_Zmol_type_sys_prep
print*,'WARNING THIS is a SYSTEM PREPARATION JOB type_prep=2'
endif
end subroutine parse_extra_line

end subroutine read_input_file



 end module parser

