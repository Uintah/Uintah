!  ANGLE TYPES
Nm_type_angles=0
i1=0; i2=0
do i = 1,lines
  NNN = the_words(i,1)%length
  do k = 1, NNN
    chtemp(k:k) = UP_CASE(the_words(i,1)%ch(k:k))
  enddo
  if (chtemp(1:NNN) == 'ANGLE_TYPES') i1 = i1 + 1
  if (chtemp(1:NNN) == 'END_ANGLE_TYPES') i2 = i2 + 1
enddo

print*, 'ANGLE_TYPES=',i1
if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  ANGLE_TYPES not equal with END_ANGLE_TYPES',i1,i2
 STOP
endif

how_many = i1

if (how_many==0) then
 print*, 'No ANGLE_TYPE was defined'
endif



IF (how_many > 0) then
allocate(where_angle_starts(how_many),where_angle_ends(how_many))
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'ANGLE_TYPES',l_skip_line,how_many,where_angle_starts,.true.)

  do i = 1, how_many
  if (NumberOfWords(where_angle_starts(i)%line) < 2) then
  write(6,*) 'ERROR in "',trim(nf),'"file : at line',where_angle_starts(i)%line, ' at least 2 records are needed',&
             'instead of just ',NumberOfWords(where_angle_starts(i)%line)
   STOP
  endif
  enddo
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'END_ANGLE_TYPES',l_skip_line,how_many,where_angle_ends,.true.)
  do i = 1, how_many
  if (where_angle_starts(i)%line >= where_angle_ends(i)%line) then
   write(6,*) 'ERROR in "',trim(nf),'"file : The description of the molecular atoms ends before it starts; ',&
              ' i.e. the key END_ANGLE_TYPE appear before the key ANGLE_TYPE ; see line ', where_angle_ends(i)%line
   STOP
  endif
  enddo

 ! map the ANGLES in molecule
 allocate(angle_2_mol(N_TYPE_MOLECULES),angle_2_mol_end(N_TYPE_MOLECULES))
 angle_2_mol=-999 ; angle_2_mol_end=-999
 do i = 1, N_TYPE_MOLECULES
 iii=0; iii1=0; iii2=0
 do j = 1, how_many
   l_1=where_angle_starts(j)%line>where_mol_starts(i)%line.and.where_angle_starts(j)%line<where_mol_ends(i)%line
   l_2=where_angle_ends(j)%line>where_mol_starts(i)%line.and.where_angle_ends(j)%line<where_mol_ends(i)%line
   if (l_1) angle_2_mol(i) = j
   if (l_2) angle_2_mol_end(i) = j

   if (l_1) then
     iii1=iii1+1
     if (iii1>1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of ANGLE_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence ANGLE_TYPE '
                STOP
     endif
   endif

   if (l_2) then
      iii2=iii2+1
      if (iii2>1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of END_ANGLE_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence END_ANGLE_TYPE '
                STOP
     endif
   endif

   if (l_1.and.l_2) then
      iii = iii + 1
   endif
   if (iii > 1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of ANGLE_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence ANGLE_TYPE ... END_ANGLE_TYPE'
                STOP
   endif
  enddo
  enddo


  do j = 1, how_many
  l_1 = .false.
  l_2 = .false.
  do i = 1, N_TYPE_MOLECULES
   l_1=l_1.or.(where_angle_starts(j)%line>where_mol_starts(i)%line.and.where_angle_starts(j)%line<where_mol_ends(i)%line)
   l_2=l_2.or.(where_angle_ends(j)%line>where_mol_starts(i)%line.and.where_angle_ends(j)%line<where_mol_ends(i)%line)
  enddo
  if (.not.l_1) then
     PRINT*, 'ERROR: in file:',trim(nf),' ANGLE_TYPES oustide MOL...ENDMOL; delete the see line:', where_angle_starts(j)%line
     STOP
  endif
  if (.not.l_1) then
     PRINT*, 'ERROR: in file:',trim(nf),' ANGLE_TYPES oustide MOL...ENDMOL; delete the see line:', where_angle_ends(j)%line
     STOP
  endif
  enddo

 do i = 1, N_TYPE_MOLECULES
  if (angle_2_mol(i) > 0 .and. angle_2_mol_end(i) < 0) then
     print*, 'ERROR: in file ',trim(nf), 'ANGLE_TYPE defined but END_ANGLE_TYPES not defined ',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and ANGLE def : see at line' ,where_angle_starts(angle_2_mol(i))%line
     STOP
  endif
  if (angle_2_mol(i) < 0 .and. angle_2_mol_end(i) > 0) then
     print*, 'ERROR: in file ',trim(nf), 'END_ANGLE_TYPE defined but ANGLE_TYPES not defined ',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and ANGLE def : see at line' ,where_angle_ends(angle_2_mol_end(i))%line
     STOP
  endif

  if (angle_2_mol(i) /= angle_2_mol_end(i)) then
    if (angle_2_mol(i)>0.and.angle_2_mol_end(i)>0) then
     print*, 'ERROR: ANGLE ... END ANGLE is not within the same MOL .. END MOL; see in infile mol def between lines',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and ANGLE def :' ,&
     where_angle_starts(angle_2_mol(i))%line,where_angle_ends(angle_2_mol_end(i))%line
     STOP
    else
     print*, 'ERROR: ANGLE definition outside any MOL definition; delete ANGLE ... endANGLE that is outside mol..end_mol'
    endif
  endif
 enddo

 iii1=0;iii2=0
 do i = 1, N_TYPE_MOLECULES
   if (angle_2_mol(i)>0) iii1 = iii1+1
   if (angle_2_mol_end(i)>0) iii2 = iii2+1
 enddo

 if (iii1 /= how_many) then
    print*, 'ERROR in file ', trim(nf),' there are ANGLE_TYPES records outside MOL...END_MOL definition. Delete them.'
    STOP
 endif
 if (iii2 /= how_many) then
    print*, 'ERROR in file ', trim(nf),' there are END_ANGLE_TYPES records outside MOL...END_MOL definition. Delete them.'
    STOP
 endif

 do i = 1, N_TYPE_MOLECULES
   if (angle_2_mol(i) > 0) then
   j = angle_2_mol(i)
   iline = where_angle_starts(j)%line
   if (NumberOfWords(iline)<2) then
      print*, 'ERROR: More records needed at line ',iline
      STOP
   endif
   NNN = the_words(iline,2)%length
   call attempt_integer_strict(NNN,the_words(iline,2)%ch(1:NNN), &
                                        Nm_type_angles(i)  , trim(nf), iline)

   endif
 enddo

ENDIF ! more than 0 angles

print*, 'Nm_type_angles=',Nm_type_angles
