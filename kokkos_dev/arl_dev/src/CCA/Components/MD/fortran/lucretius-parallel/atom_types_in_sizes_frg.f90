i1=0; i2=0
do i = 1,lines
  NNN = the_words(i,1)%length
  do k = 1,NNN
  chtemp(k:k) = UP_CASE(the_words(i,1)%ch(k:k))
  enddo
  if (chtemp(1:NNN) == 'ATOM_TYPES') i1 = i1 + 1
  if (chtemp(1:NNN) == 'END_ATOM_TYPES') i2 = i2 + 1
enddo
if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  ATOM_TYPE not equal with END_ATOM_TYPES',i1,i2
 STOP
endif

how_many = i1

print*, 'ATOM_TYPES=',i1

if (how_many==0) then
 print*, 'ERROR:: in the file:',trim(nf), ' No ATOM_TYPE was defined'
 STOP
endif

if (i1 /= N_TYPE_MOLECULES) then
 PRINT*, 'ERROR: in file:',trim(nf),' Mismatch between the number of ATOM_TYPES and MOLECULE_TYPE; they should be equal; i1 N_TYPE_MOLECULES=', i1 , N_TYPE_MOLECULES
 STOP
endif

allocate(where_atom_starts(N_TYPE_MOLECULES),where_atom_ends(N_TYPE_MOLECULES))
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'ATOM_TYPES',l_skip_line,N_TYPE_MOLECULES,where_atom_starts,.true.)
  do i = 1, N_TYPE_MOLECULES
  if (NumberOfWords(where_atom_starts(i)%line) < 2) then
  write(6,*) 'ERROR in "',trim(nf),'"file : at line',where_atom_starts(i)%line, ' at least 2 records are needed',&
              'instead of just ',NumberOfWords(where_atom_starts(i)%line)
   STOP
  endif
   iline = where_atom_starts(i)%line
   NNN = the_words(iline,2)%length
   call attempt_integer_strict(NNN,the_words(iline,2)%ch(1:NNN), &
                                        N_type_atoms_per_mol_type(i)  , trim(nf), iline)

  enddo
print*, 'N_type_atoms_per_mol_type=',N_type_atoms_per_mol_type
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'END_ATOM_TYPES',l_skip_line,N_TYPE_MOLECULES,where_atom_ends,.true.)
  do i = 1, N_TYPE_MOLECULES
  if (where_atom_starts(i)%line >= where_atom_ends(i)%line) then
   write(6,*) 'ERROR in "',trim(nf),'"file : The description of the molecular atoms ends before it starts; ',&
              ' i.e. the key END_ATOM_TYPE appear before the key ATOM_TYPE ; see line ', where_atom_ends(i)%line
   STOP
  endif
  enddo


allocate(atom_2_mol(N_TYPE_MOLECULES))
atom_2_mol=-999
do i = 1, N_TYPE_MOLECULES
iii=0
j=i
   l_1=where_atom_starts(j)%line>where_mol_starts(i)%line.and.where_atom_starts(j)%line<where_mol_ends(i)%line
   l_2=where_atom_ends(j)%line>where_mol_starts(i)%line.and.where_atom_ends(j)%line<where_mol_ends(i)%line

   if ((l_1.and.(.not.l_2)).or.((.not.l_1).and.l_2) )   then
       PRINT*, 'ERROR: in file:',trim(nf),' Mismatch between the number of ATOM_TYPES and MOLECULE_TYPES; ',&
           ' The definition of a certain bond ATOM_TYPE ... END_ATOM_TYPE must be contained within the definition',&
            ' of a certain molecule MOLECULE_TYPE..END_MOLECULE_TYPE; see the molecular type in input file at lines:',&
                   where_mol_starts(i),where_mol_ends(i),'end atom types between lines :',where_atom_starts(j),where_atom_ends(j)
           STOP
   endif
   if (l_1.and.l_2) atom_2_mol(j)=i
enddo

do i = 1, N_TYPE_MOLECULES
iii=0
do j = 1, N_TYPE_MOLECULES
   l_1=where_atom_starts(j)%line>where_mol_starts(i)%line.and.where_atom_starts(j)%line<where_mol_ends(i)%line
   l_2=where_atom_ends(j)%line>where_mol_starts(i)%line.and.where_atom_ends(j)%line<where_mol_ends(i)%line

   if (l_1.and.l_2) then
      iii = iii + 1
   endif
   if (iii > 1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of ATOM_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line ,&
                 'and delete a sequence ATOM_TYPE ... END_ATOM_TYPE'
                STOP
   endif
enddo
enddo

do i = 1, N_TYPE_MOLECULES
    if (atom_2_mol(i) < 0) then
          PRINT*, 'ERROR: in file:',trim(nf),' An ATOM_TYPE definition is outside the MOLECULE_TYPE..END_MOLECULE_TYPE'
          print*, 'Verify if the entries from the file:',trim(nf),'between the lines: ', where_atom_starts(i),where_atom_ends(i),&
                  ' should not be assigned to a certain molecule ',&
                  ' i.e. within a MOLECULE_TYPE..END_MOLECULE_TYPE sequence. If not: comment or delete between ',&
                           where_atom_starts(i),where_atom_ends(i)
          STOP
    endif
enddo
print*, 'atom_2_mol=',atom_2_mol


