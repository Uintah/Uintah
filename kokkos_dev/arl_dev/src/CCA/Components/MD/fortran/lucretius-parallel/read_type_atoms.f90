
 subroutine read_type_atoms(imol, i1,i_start, i_end, many, NumberOfWords, Max_words_per_line, the_words, &
                            ff_hit, where_in_file, nf)
 use atom_type_data
 use mol_type_data
 use max_sizes_data, only : Max_Atom_Name_Len
 use force_field_data, only : predef_ff_atom, N_predef_ff_atoms, N_predef_ff_dummies, predef_ff_dummy
 use CTRLs_data
 use types_module, only : word_type
 use field_constrain_data, only : is_type_atom_field_constrained
 use chars, only : locate_word_in_key,attempt_integer_strict,attempt_real_strict, locate_UPCASE_word_in_key

 implicit none
 integer, intent(IN) :: imol                            
 integer, intent( inout) :: i1 ! i1 is the actual type of the atom
 integer, intent(in) :: i_start, i_end, Max_words_per_line
 type(word_type),intent(in) :: the_words(i_start:i_end,1:Max_words_per_line)
 integer, intent(IN) :: NumberOfWords(i_start:i_end), where_in_file(i_start:i_end)
 character(*) , intent(IN) :: nf
 integer, intent(INOUT) :: many(:)
 logical, intent(INOUT) :: ff_hit(:)
 real(8) bla
 logical l_OK, l_found, l_1,l_found1
 integer i,j,k,kkk, i10, iii, sum_many, NNN, jjj, ijk
 

 sum_many = 0
   do j = i_start, i_end
     i1 = i1 + 1
     if (NumberOfWords(j) < 2) then
        write(6,*) 'ERROR in the file"',trim(nf),'" at the line ',j, &
                  'there must be at least 6 records and you only have ',&
                   NumberOfWords(j)
        STOP
     endif

     many(i1) = 1
     call locate_word_in_key('many',1,the_words(j,:), l_found,kkk) !
     if (l_found) then
        call attempt_integer_strict(the_words(j,kkk+1)%length,&
             the_words(j,kkk+1)%ch(1:the_words(j,kkk+1)%length), &
             many(i1), trim(nf), where_in_file(j))
     endif
     sum_many = sum_many + many(i1)

      if (sum_many > N_type_atoms_per_mol_type(imol)) then
   print*, 'ERROR too many atoms defined in file: ',trim(nf),' between lines ',&
   where_in_file(i_start), where_in_file(i_end)
   print*, 'Either reduce the number of atom definitions or increase the integer number at line',where_in_file(j)
   print*, imol,'sum_many _type_atoms_per_mol_type(imol)=',sum_many, N_type_atoms_per_mol_type(imol)
   STOP
      endif
!   atom_type_molindex
     NNN = the_words(j,1)%length
     call attempt_integer_strict(the_words(j,1)%length,&
             the_words(j,1)%ch(1:the_words(j,1)%length), &
             atom_type_molindex(i1), trim(nf), where_in_file(j))

! name
     NNN = the_words(j,2)%length
     atom_type_name(i1)(1:Max_Atom_Name_Len) = ' '
     if (NNN > Max_Atom_Name_Len) then
         print*,'WARNING: in file:',trim(nf),' at line',where_in_file(j),' the name of the atom will be truncated'
     endif
     do jjj = 1,min(NNN,Max_Atom_Name_Len)
           atom_type_name(i1)(jjj:jjj)= the_words(j,2)%ch(jjj:jjj)
     enddo
!  identify the atom if it is in the force field list; if not STOP
     l_OK = .false.
     do jjj = 1, N_predef_ff_atoms
       if (trim(predef_ff_atom(jjj)%name) == trim(atom_type_name(i1)) ) then
         map_atom_type_to_predef_atom_ff(i1) = jjj
         ff_hit(jjj) = .true.
         l_OK = .true.
       endif
     enddo ! jjj
     if (.not.l_OK) then
       print*, 'The atom from  the molecular description from line ',where_in_file(j),' in file:',trim(nf),&
              ' does not have any force field definition. either define it in the sequence ',&
              ' DEFINE_ATOMS .. END_DEFINE_ATOMS or remove it; for safety STOP'
       STOP
     endif
! dumy
! dumy
     jjj = map_atom_type_to_predef_atom_ff(i1)
     atom_type_mass(i1) = predef_ff_atom(jjj)%mass
     atom_type_isDummy(i1) =predef_ff_atom(jjj)%is_dummy
     atom_type_DummyInfo(i1)%i = predef_ff_atom(jjj)%dummy%GeomType
     atom_type_DummyInfo(i1)%r(1:3) = predef_ff_atom(jjj)%dummy%the_params(1:3)
     call locate_UPCASE_word_in_key('DMY',4,the_words(j,:), l_found,kkk) !
     if (l_found) then
        atom_type_mass(i1) = 0.0d0
        print*, 'DUMMY atom defined in in_file at line:',where_in_file(j),' The mass of the atom will be reset to ZERO'
        atom_type_isDummy(i1) =.true.
        k=1
            call attempt_integer_strict(the_words(j,kkk+k)%length,&
                 the_words(j,kkk+k)%ch(1:the_words(j,kkk+k)%length), &
                 atom_type_DummyInfo(i1)%i, trim(nf), where_in_file(j))
        do k = 2, 4
            call attempt_real_strict(the_words(j,kkk+k)%length,&
                 the_words(j,kkk+k)%ch(1:the_words(j,kkk+k)%length), &
                 atom_type_DummyInfo(i1)%r(k-1), trim(nf), where_in_file(j))

        enddo 
     endif
     call locate_UPCASE_word_in_key('*DMY',1,the_words(j,:), l_found1,kkk) !
     if (l_found.and.l_found1) then
        print*, 'ERROR in file ',trim(nf), ' at line ',where_in_file(j), ' DMY and *DMY cannot be in the same line'
        STOP
     endif 
     if (l_found1) then
         l_1 = .false.
         NNN = the_words(j,kkk+1)%length
         do k = 1, N_predef_ff_dummies
           if(the_words(j,kkk+1)%ch(1:NNN)==trim(predef_ff_dummy(k)%name)) then
               l_1 = .true.
               atom_type_DummyInfo(i1)%i = predef_ff_dummy(k)%GeomType
               atom_type_DummyInfo(i1)%r(1:3) =predef_ff_dummy(k)%the_params
               atom_type_isDummy(i1) =.true.
           endif
           if (.not.l_1) then
              print*, 'ERROR in file ',trim(nf),' at line ',where_in_file(j), ' dummy type not predefined in Define_dummy...End_define_dummy'
              STOP
           endif
         enddo
     endif 
! is sfc?
     jjj = map_atom_type_to_predef_atom_ff(i1)
     if (IS_TYPE_ATOM_FIELD_CONSTRAINED(i1)) then
     if (sfc_already_defined(i1)) then
        bla= predef_ff_atom(jjj)%sfc
     endif
     atom_type_sfield_CONSTR(i1) = predef_ff_atom(jjj)%sfc
     if (sfc_already_defined(i1)) then
       print*, 'OVERWRITE: sfc defined in mol_type definition as:',bla,' redefined  in ff style_atom definition as:',&
       atom_type_sfield_CONSTR(i1)
     endif
     else
       IS_TYPE_ATOM_FIELD_CONSTRAINED(i1) = predef_ff_atom(jjj)%isQsfc
     endif ! IS_TYPE_ATOM_FIELD_CONSTRAINED(i1)
     call locate_UPCASE_word_in_key('*SFC',1,the_words(j,:), l_found,kkk)
!     l_ANY_S_FIELD_CONS_CTRL=.false. ! Already initialized at mols.
     if  (l_found) then
     l_ANY_S_FIELD_CONS_CTRL=.true.
       if (sfc_already_defined(i1)) then
          bla = atom_type_sfield_CONSTR(i1)
       endif
       IS_TYPE_ATOM_FIELD_CONSTRAINED(i1) = .true.
          call attempt_real_strict(the_words(j,kkk+1)%length,&
           the_words(j,kkk+1)%ch(1:the_words(j,kkk+1)%length), &
           atom_type_sfield_CONSTR(i1) , trim(nf), where_in_file(j))
       print*, 'OVERWRITE: sfc constraint redefined for the atom at line :',where_in_file(j),' in file :',trim(nf)
       if (sfc_already_defined(i1)) then
          bla = atom_type_sfield_CONSTR(i1)
          print*,'OVERWRITE: sfc defined in mol_type definition as:',bla,' redefined  at line :',&
                 where_in_file(j),' in file :',trim(nf), ' as ',atom_type_sfield_CONSTR(i1) 
       endif
     endif
!  actual Q
     atom_type_charge(i1) = predef_ff_atom(jjj)%Q
     call locate_UPCASE_word_in_key('CH',1,the_words(j,:), l_found,kkk)
     if  (l_found) then
          call attempt_real_strict(the_words(j,kkk+1)%length,&
           the_words(j,kkk+1)%ch(1:the_words(j,kkk+1)%length), &
           atom_type_charge(i1) , trim(nf), where_in_file(j))
       print*, ' charge redefined for the atom at line :',where_in_file(j),' in file :',trim(nf)
     endif
     if (dabs(atom_type_charge(i1)) > 1.0d-9) l_ANY_Q_CTRL =.true.;
     is_type_charge_distributed(i1) = predef_ff_atom(jjj)%isQdistributed
!  Qpol
     atom_type_Q_pol(i1) = predef_ff_atom(jjj)%Qpol
!     call locate_UPCASE_word_in_key('*CH',1,the_words(j,:), l_found,kkk)
!     if  (l_found) then
!          call attempt_real_strict(the_words(j,kkk+1)%length,&
!           the_words(j,kkk+1)%ch(1:the_words(j,kkk+1)%length), &
!           atom_type_Q_pol(i1) , trim(nf), where_in_file(j))
!       print*, ' charge polarization redefined for the atom at line :',where_in_file(j),' in file :',trim(nf)
!     endif

     is_type_charge_pol(i1) = atom_type_Q_pol(i1) > 1.0d-8
     if (is_type_charge_pol(i1)) l_ANY_Q_POL_CTRL=.true.

     if (atom_type_Q_pol(i1) >  1.0d-10) then
     if(is_type_charge_distributed(i1)) then
          l_ANY_QG_POL_CTRL=.true.    ;    l_ANY_Q_POL_CTRL=.true.
     else
          l_ANY_QP_POL_CTRL=.true.    ;    l_ANY_Q_POL_CTRL=.true.
     endif
     endif
!  actual Dip
     atom_type_dipol(i1)  = predef_ff_atom(jjj)%dip
     atom_type_DIR_dipol(i1,1:3) = predef_ff_atom(jjj)%dipDir(1:3)
     call locate_UPCASE_word_in_key('DIP',1,the_words(j,:), l_found,kkk)
     if  (l_found) then
          call attempt_real_strict(the_words(j,kkk+1)%length,&
           the_words(j,kkk+1)%ch(1:the_words(j,kkk+1)%length), &
           atom_type_dipol(i1) , trim(nf), where_in_file(j))
       print*, ' dipole redefined for the atom at line :',where_in_file(j),' in file :',trim(nf)
     endif
     if (dabs(atom_type_charge(i1)) > 1.0d-9) l_ANY_Q_CTRL =.true.;
     if (dabs(atom_type_dipol(i1)) > 1.0d-10) l_ANY_DIPOLE_CTRL=.true.
     call locate_UPCASE_word_in_key('DIPDIR',3,the_words(j,:), l_found,kkk)
     if  (l_found) then
     do ijk=1,3
          call attempt_real_strict(the_words(j,kkk+ijk)%length,&
           the_words(j,kkk+ijk)%ch(1:the_words(j,kkk+ijk)%length), &
           atom_type_DIR_dipol(i1,ijk) , trim(nf), where_in_file(j))
     enddo
       print*, ' dipole direction redefined for the atom at line :',where_in_file(j),' in file :',trim(nf)
     endif
     if (dabs(atom_type_charge(i1)) > 1.0d-9) l_ANY_Q_CTRL =.true.;
     if (dabs(atom_type_dipol(i1)) > 1.0d-10) l_ANY_DIPOLE_CTRL=.true.
!  DipPol
     atom_type_DIPOLE_pol(i1) = predef_ff_atom(jjj)%dipPol
     is_type_dipole_pol(i1)   = predef_ff_atom(jjj)%isdipPol

!     call locate_UPCASE_word_in_key('*DIP',1,the_words(j,:), l_found,kkk)
!     if  (l_found) then
!          call attempt_real_strict(the_words(j,kkk+1)%length,&
!           the_words(j,kkk+1)%ch(1:the_words(j,kkk+1)%length), &
!           atom_type_DIPOLE_pol(i1) , trim(nf), where_in_file(j))
!       print*, ' dipole redefined for the atom at line :',where_in_file(j),' in file :',trim(nf)
!     endif


     is_type_dipole_pol(i1) = dabs(atom_type_DIPOLE_pol(i1)) > 1.0d-8
     if (is_type_dipole_pol(i1)) l_ANY_DIPOLE_POL_CTRL=.true.
!  WALL WALL_1
     l_TYPE_ATOM_WALL(i1) = predef_ff_atom(jjj)%isWALL
     l_TYPE_ATOM_WALL_1(i1) = predef_ff_atom(jjj)%isWALL_1
     call locate_UPCASE_word_in_key('WALL',0,the_words(j,:), l_found,kkk)
    if (l_found) then
       l_TYPE_ATOM_WALL(i1) = .true.
       print*, 'WALL character on atom at line',where_in_file(j),' was redefined vs ff.'
    endif
    call locate_word_in_key('WALL_1',0,the_words(j,:), l_found,kkk)
    if (l_found) then
       l_TYPE_ATOM_WALL_1(i1) = .true.
       print*, 'WALL_1 character on atom at line',where_in_file(j),' was redefined vs ff.'
    endif
    if (l_TYPE_ATOM_WALL(i1).and.l_TYPE_ATOM_WALL_1(i1)) then
       print*, 'ERROR: in file:',trim(nf),' at line',where_in_file(j),' CANNOT have an atom both l_WALL and l_WALL1 '
       STOP
    endif
    l_TYPEatom_do_stat_on_type(i1) = predef_ff_atom(jjj)%more_logic(1)
    call locate_UPCASE_word_in_key('+STAT',0,the_words(j,:), l_found,kkk)
    if (l_found) then
       l_TYPEatom_do_stat_on_type(i1) = .true.
    endif

 i10 = i1
 do iii = 2,many(i1)
    i1 = i1 + 1
    many(i1) = 0
    atom_type_molindex(i1) = atom_type_molindex(i10)
    atom_type_name(i1)     = atom_type_name(i10)
    map_atom_type_to_predef_atom_ff(i1) = map_atom_type_to_predef_atom_ff(i10)
    atom_type_isDummy(i1)  = atom_type_isDummy(i10)
    atom_type_mass(i1)     = atom_type_mass(i10)
    atom_type_DummyInfo(i1)%i = atom_type_DummyInfo(i10)%i
    atom_type_DummyInfo(i1)%r = atom_type_DummyInfo(i10)%r
    IS_TYPE_ATOM_FIELD_CONSTRAINED(i1) = IS_TYPE_ATOM_FIELD_CONSTRAINED(i10)
    atom_type_sfield_CONSTR(i1)        = atom_type_sfield_CONSTR(i10)
    atom_type_charge(i1)               = atom_type_charge(i10)
    is_type_charge_distributed(i1)     = is_type_charge_distributed(i10)
!        if (is_type_charge_distributed(i1)) then
!         atom_type_1GAUSS_charge(i1) = atom_type_charge(i1)
!        endif
!    atom_type_1GAUSS_charge_distrib(i1) = atom_type_1GAUSS_charge_distrib(i10)
    atom_type_Q_pol(i1)         = atom_type_Q_pol(i10)
    is_type_charge_pol(i1)      = is_type_charge_pol(i10)
    atom_type_dipol(i1)         = atom_type_dipol(i10)
    atom_type_DIR_dipol(i1,1:3) = atom_type_DIR_dipol(i10,1:3)
    atom_type_DIPOLE_pol(i1)    = atom_type_DIPOLE_pol(i10)
    is_type_dipole_pol(i1)      = is_type_dipole_pol(i10)
    l_TYPE_ATOM_WALL(i1)        = l_TYPE_ATOM_WALL(i10)
    l_TYPE_ATOM_WALL_1(i1)      = l_TYPE_ATOM_WALL_1(i10)
    l_TYPEatom_do_stat_on_type(i1)  = l_TYPEatom_do_stat_on_type(i10)
 enddo



 enddo ! j
 if (sum_many < N_type_atoms_per_mol_type(imol)) then
   print*, 'ERROR too few atoms defined in file: ',trim(nf),' between lines ',&
   where_in_file(i_start), where_in_file(i_end)
   print*, 'Either increase the number of atom definitions or decrease the integer number above the line',where_in_file(i_start), &
   ' where ATOM_TYPES is defined'
   STOP
 endif
 j = 0
 do i = 1, N_type_atoms
  if (l_TYPEatom_do_stat_on_type(i)) j = j + 1
 enddo
 print*, 'predef_ff_atom(jjj)%more_logic(1)=',predef_ff_atom(:)%more_logic(1)
 print*, j,'statistics on atom types : ',l_TYPEatom_do_stat_on_type


  


! print*, 'exit from read_type_atoms'
 end subroutine read_type_atoms
