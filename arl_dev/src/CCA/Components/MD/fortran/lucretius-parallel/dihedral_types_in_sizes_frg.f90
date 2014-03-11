!  DIHEDRAL TYPES
Nm_type_dihedrals=0
i1=0; i2=0
do i = 1,lines
  NNN = the_words(i,1)%length
  do k = 1, NNN
    chtemp(k:k) = UP_CASE(the_words(i,1)%ch(k:k))
  enddo
  if (chtemp(1:NNN) == 'DIHEDRAL_TYPES') i1 = i1 + 1
  if (chtemp(1:NNN) == 'END_DIHEDRAL_TYPES') i2 = i2 + 1
enddo

print*, 'DIHEDRAL_TYPES=',i1
if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  DIHEDRAL_TYPES not equal with END_DIHEDRAL_TYPES',i1,i2
 STOP
endif

how_many = i1

if (how_many==0) then
 print*, 'No DIHEDRAL_TYPE was defined'
endif



IF (how_many > 0) then
allocate(where_DIHEDRAL_starts(how_many),where_DIHEDRAL_ends(how_many))
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'DIHEDRAL_TYPES',l_skip_line,how_many,where_DIHEDRAL_starts,.true.)

  do i = 1, how_many
  if (NumberOfWords(where_DIHEDRAL_starts(i)%line) < 2) then
  write(6,*) 'ERROR in "',trim(nf),'"file : at line',where_DIHEDRAL_starts(i)%line, ' at least 2 records are needed',&
             'instead of just ',NumberOfWords(where_DIHEDRAL_starts(i)%line)
   STOP
  endif
  enddo
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'END_DIHEDRAL_TYPES',l_skip_line,how_many,where_DIHEDRAL_ends,.true.)
  do i = 1, how_many
  if (where_DIHEDRAL_starts(i)%line >= where_DIHEDRAL_ends(i)%line) then
   write(6,*) 'ERROR in "',trim(nf),'"file : The description of the molecular atoms ends before it starts; ',&
              ' i.e. the key END_DIHEDRAL_TYPE appear before the key DIHEDRAL_TYPE ; see line ', where_DIHEDRAL_ends(i)%line
   STOP
  endif
  enddo


 ! map the DIHEDRALS in molecule
 allocate(DIHEDRAL_2_mol(N_TYPE_MOLECULES),DIHEDRAL_2_mol_end(N_TYPE_MOLECULES))
 DIHEDRAL_2_mol=-999 ; DIHEDRAL_2_mol_end=-999
 do i = 1, N_TYPE_MOLECULES
 iii=0; iii1=0; iii2=0
 do j = 1, how_many
   l_1=where_DIHEDRAL_starts(j)%line>where_mol_starts(i)%line.and.where_DIHEDRAL_starts(j)%line<where_mol_ends(i)%line
   l_2=where_DIHEDRAL_ends(j)%line>where_mol_starts(i)%line.and.where_DIHEDRAL_ends(j)%line<where_mol_ends(i)%line
   if (l_1) DIHEDRAL_2_mol(i) = j
   if (l_2) DIHEDRAL_2_mol_end(i) = j

   if (l_1) then
     iii1=iii1+1
     if (iii1>1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of DIHEDRAL_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence DIHEDRAL_TYPE '
                STOP
     endif
   endif

   if (l_2) then
      iii2=iii2+1
      if (iii2>1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of END_DIHEDRAL_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence END_DIHEDRAL_TYPE '
                STOP
     endif
   endif

   if (l_1.and.l_2) then
      iii = iii + 1
   endif
   if (iii > 1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of DIHEDRAL_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence DIHEDRAL_TYPE ... END_DIHEDRAL_TYPE'
                STOP
   endif
  enddo
  enddo


  do j = 1, how_many
  l_1 = .false.
  l_2 = .false.
  do i = 1, N_TYPE_MOLECULES
   l_1=l_1.or.(where_DIHEDRAL_starts(j)%line>where_mol_starts(i)%line.and.where_DIHEDRAL_starts(j)%line<where_mol_ends(i)%line)
   l_2=l_2.or.(where_DIHEDRAL_ends(j)%line>where_mol_starts(i)%line.and.where_DIHEDRAL_ends(j)%line<where_mol_ends(i)%line)
  enddo
  if (.not.l_1) then
     PRINT*, 'ERROR: in file:',trim(nf),' DIHEDRAL_TYPES oustide MOL...ENDMOL; delete the see line:', where_DIHEDRAL_starts(j)%line
     STOP
  endif
  if (.not.l_1) then
     PRINT*, 'ERROR: in file:',trim(nf),' DIHEDRAL_TYPES oustide MOL...ENDMOL; delete the see line:', where_DIHEDRAL_ends(j)%line
     STOP
  endif
  enddo

 do i = 1, N_TYPE_MOLECULES
  if (DIHEDRAL_2_mol(i) > 0 .and. DIHEDRAL_2_mol_end(i) < 0) then
     print*, 'ERROR: in file ',trim(nf), 'DIHEDRAL_TYPE defined but END_DIHEDRAL_TYPES not defined ',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and DIHEDRAL def : see at line' ,where_DIHEDRAL_starts(DIHEDRAL_2_mol(i))%line
     STOP
  endif
  if (DIHEDRAL_2_mol(i) < 0 .and. DIHEDRAL_2_mol_end(i) > 0) then
     print*, 'ERROR: in file ',trim(nf), 'END_DIHEDRAL_TYPE defined but DIHEDRAL_TYPES not defined ',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and DIHEDRAL def : see at line' ,where_DIHEDRAL_ends(DIHEDRAL_2_mol_end(i))%line
     STOP
  endif

  if (DIHEDRAL_2_mol(i) /= DIHEDRAL_2_mol_end(i)) then
    if (DIHEDRAL_2_mol(i)>0.and.DIHEDRAL_2_mol_end(i)>0) then
     print*, 'ERROR: DIHEDRAL ... END DIHEDRAL is not within the same MOL .. END MOL; see in infile mol def between lines',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and DIHEDRAL def :' ,&
     where_DIHEDRAL_starts(DIHEDRAL_2_mol(i))%line,where_DIHEDRAL_ends(DIHEDRAL_2_mol_end(i))%line
     STOP
    else
     print*, 'ERROR: DIHEDRAL definition outside any MOL definition; delete DIHEDRAL ... endDIHEDRAL that is outside mol..end_mol'
    endif
  endif
 enddo

 iii1=0;iii2=0
 do i = 1, N_TYPE_MOLECULES
   if (DIHEDRAL_2_mol(i)>0) iii1 = iii1+1
   if (DIHEDRAL_2_mol_end(i)>0) iii2 = iii2+1
 enddo

 if (iii1 /= how_many) then
    print*, 'ERROR in file ', trim(nf),' there are DIHEDRAL_TYPES records outside MOL...END_MOL definition. Delete them.'
    STOP
 endif
 if (iii2 /= how_many) then
    print*, 'ERROR in file ', trim(nf),' there are END_DIHEDRAL_TYPES records outside MOL...END_MOL definition. Delete them.'
    STOP
 endif

 do i = 1, N_TYPE_MOLECULES
   if (dihedral_2_mol(i) > 0) then
   j = dihedral_2_mol(i)
   iline = where_dihedral_starts(j)%line
   if (NumberOfWords(iline)<2) then
      print*, 'ERROR: More records needed at line ',iline
      STOP
   endif
   NNN = the_words(iline,2)%length
   call attempt_integer_strict(NNN,the_words(iline,2)%ch(1:NNN), &
                                        Nm_type_dihedrals(i)  , trim(nf), iline)

   endif
 enddo


ENDIF ! more than 0 DIHEDRALs

print*,'Nm_type_dihedrals=',Nm_type_dihedrals
