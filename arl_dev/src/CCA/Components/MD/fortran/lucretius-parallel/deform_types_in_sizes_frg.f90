!  Deform TYPES
Nm_type_deforms=0
i1=0; i2=0
do i = 1,lines
  NNN = the_words(i,1)%length
  do k = 1, NNN
    chtemp(k:k) = UP_CASE(the_words(i,1)%ch(k:k))
  enddo
  if (chtemp(1:NNN) == 'DEFORM_TYPES') i1 = i1 + 1
  if (chtemp(1:NNN) == 'END_DEFORM_TYPES') i2 = i2 + 1
enddo

print*, 'DEFORM_TYPES=',i1,i2
if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  DEFORM_TYPES not equal with END_DEFORM_TYPES',i1,i2
 STOP
endif

how_many = i1

if (how_many==0) then
 print*, 'No DEFORM_TYPE was defined'
endif



IF (how_many > 0) then
allocate(where_DEFORM_starts(how_many),where_DEFORM_ends(how_many))
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'DEFORM_TYPES',l_skip_line,how_many,where_DEFORM_starts,.true.)

  do i = 1, how_many
  if (NumberOfWords(where_DEFORM_starts(i)%line) < 2) then
  write(6,*) 'ERROR in "',trim(nf),'"file : at line',where_DEFORM_starts(i)%line, ' at least 2 records are needed',&
             'instead of just ',NumberOfWords(where_DEFORM_starts(i)%line)
   STOP
  endif
  enddo
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'END_DEFORM_TYPES',l_skip_line,how_many,where_DEFORM_ends,.true.)
  do i = 1, how_many
  if (where_DEFORM_starts(i)%line >= where_DEFORM_ends(i)%line) then
   write(6,*) 'ERROR in "',trim(nf),'"file : The description of the molecular atoms ends before it starts; ',&
              ' i.e. the key END_OUTOFPLANEDEFORMS_TYPES appear before the key OUTOFPLANEDEFORMS_TYPES ; see line ', &
              where_DEFORM_ends(i)%line
   STOP
  endif
  enddo


 ! map the DEFORMS in molecule
 allocate(DEFORM_2_mol(N_TYPE_MOLECULES),DEFORM_2_mol_end(N_TYPE_MOLECULES))
 DEFORM_2_mol=-999 ; DEFORM_2_mol_end=-999
 do i = 1, N_TYPE_MOLECULES
 iii=0; iii1=0; iii2=0
 do j = 1, how_many
   l_1=where_DEFORM_starts(j)%line>where_mol_starts(i)%line.and.where_DEFORM_starts(j)%line<where_mol_ends(i)%line
   l_2=where_DEFORM_ends(j)%line>where_mol_starts(i)%line.and.where_DEFORM_ends(j)%line<where_mol_ends(i)%line
   if (l_1) DEFORM_2_mol(i) = j
   if (l_2) DEFORM_2_mol_end(i) = j

   if (l_1) then
     iii1=iii1+1
     if (iii1>1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of DEFORM_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence DEFORM_TYPE '
                STOP
     endif
   endif

   if (l_2) then
      iii2=iii2+1
      if (iii2>1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of END_DEFORM_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence END_DEFORM_TYPE '
                STOP
     endif
   endif

   if (l_1.and.l_2) then
      iii = iii + 1
   endif
   if (iii > 1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of DEFORM_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence DEFORM_TYPE ... END_DEFORM_TYPE'
                STOP
   endif
  enddo
  enddo


  do j = 1, how_many
  l_1 = .false.
  l_2 = .false.
  do i = 1, N_TYPE_MOLECULES
   l_1=l_1.or.(where_DEFORM_starts(j)%line>where_mol_starts(i)%line.and.where_DEFORM_starts(j)%line<where_mol_ends(i)%line)
   l_2=l_2.or.(where_DEFORM_ends(j)%line>where_mol_starts(i)%line.and.where_DEFORM_ends(j)%line<where_mol_ends(i)%line)
  enddo
  if (.not.l_1) then
     PRINT*, 'ERROR: in file:',trim(nf),' DEFORM_TYPES oustide MOL...ENDMOL; delete the see line:', where_DEFORM_starts(j)%line
     STOP
  endif
  if (.not.l_1) then
     PRINT*, 'ERROR: in file:',trim(nf),' DEFORM_TYPES oustide MOL...ENDMOL; delete the see line:', where_DEFORM_ends(j)%line
     STOP
  endif
  enddo

 do i = 1, N_TYPE_MOLECULES
  if (DEFORM_2_mol(i) > 0 .and. DEFORM_2_mol_end(i) < 0) then
     print*, 'ERROR: in file ',trim(nf), 'DEFORM_TYPE defined but END_DEFORM_TYPES not defined ',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and DEFORM def : see at line' ,where_DEFORM_starts(DEFORM_2_mol(i))%line
     STOP
  endif
  if (DEFORM_2_mol(i) < 0 .and. DEFORM_2_mol_end(i) > 0) then
     print*, 'ERROR: in file ',trim(nf), 'END_DEFORM_TYPE defined but DEFORM_TYPES not defined ',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and DEFORM def : see at line' ,where_DEFORM_ends(DEFORM_2_mol_end(i))%line
     STOP
  endif

  if (DEFORM_2_mol(i) /= DEFORM_2_mol_end(i)) then
    if (DEFORM_2_mol(i)>0.and.DEFORM_2_mol_end(i)>0) then
     print*, 'ERROR: DEFORM ... END DEFORM is not within the same MOL .. END MOL; see in infile mol def between lines',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and DEFORM def :' ,&
     where_DEFORM_starts(DEFORM_2_mol(i))%line,where_DEFORM_ends(DEFORM_2_mol_end(i))%line
     STOP
    else
     print*, 'ERROR: DEFORM definition outside any MOL definition; delete DEFORM ... endDEFORM that is outside mol..end_mol'
    endif
  endif
 enddo

 iii1=0;iii2=0
 do i = 1, N_TYPE_MOLECULES
   if (DEFORM_2_mol(i)>0) iii1 = iii1+1
   if (DEFORM_2_mol_end(i)>0) iii2 = iii2+1
 enddo

 if (iii1 /= how_many) then
    print*, 'ERROR in file ', trim(nf),' there are DEFORM_TYPES records outside MOL...END_MOL definition. Delete them.'
    STOP
 endif
 if (iii2 /= how_many) then
    print*, 'ERROR in file ', trim(nf),' there are END_DEFORM_TYPES records outside MOL...END_MOL definition. Delete them.'
    STOP
 endif

 do i = 1, N_TYPE_MOLECULES
   if (deform_2_mol(i) > 0) then
   j = deform_2_mol(i)
   iline = where_DEFORM_starts(j)%line
   if (NumberOfWords(iline)<2) then
      print*, 'ERROR: More records needed at line ',iline
      STOP
   endif
   NNN = the_words(iline,2)%length
   call attempt_integer_strict(NNN,the_words(iline,2)%ch(1:NNN), &
                                        Nm_type_deforms(i)  , trim(nf), iline)

   endif
 enddo


ENDIF ! more than 0 DEFORMs

print*,'Nm_type_deforms=',Nm_type_deforms
