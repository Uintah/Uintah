!  BOND TYPES
Nm_type_bonds=0
Nm_type_constrains = 0

i1=0; i2=0
do i = 1,lines
  NNN = the_words(i,1)%length
  do k = 1, NNN
    chtemp(k:k) = UP_CASE(the_words(i,1)%ch(k:k))
  enddo
  if (chtemp(1:NNN) == 'BOND_TYPES') i1 = i1 + 1
  if (chtemp(1:NNN) == 'END_BOND_TYPES') i2 = i2 + 1
enddo

print*, 'BOND_TYPES=',i1
if (i1/=i2) then
 print*, 'ERROR in the file:',trim(nf), ' number of keys  BOND_TYPES not equal with END_BOND_TYPES',i1,i2
 STOP
endif

how_many = i1

if (how_many==0) then
 print*, 'No BOND_TYPE was defined'
endif

print*, 'how many bonds =',how_many



IF (how_many > 0) then
allocate(where_bond_starts(how_many),where_bond_ends(how_many))
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'BOND_TYPES',l_skip_line,how_many,where_bond_starts,.true.)
print*,'11:',where_bond_starts%line
  do i = 1, how_many
  if (NumberOfWords(where_bond_starts(i)%line) < 2) then
  write(6,*) 'ERROR in "',trim(nf),'"file : at line',where_bond_starts(i)%line, ' at least 2 records are needed',&
             'instead of just ',NumberOfWords(where_bond_starts(i)%line)
   STOP
  endif
  enddo
  call search_words_gi(1,lines,lines,trim(nf),Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                   'END_BOND_TYPES',l_skip_line,how_many,where_bond_ends,.true.)
  do i = 1, how_many
  if (where_bond_starts(i)%line >= where_bond_ends(i)%line) then
   write(6,*) 'ERROR in "',trim(nf),'"file : The description of the molecular atoms ends before it starts; ',&
              ' i.e. the key END_BOND_TYPE appear before the key BOND_TYPE ; see line ', where_bond_ends(i)%line
   STOP
  endif

  enddo

print*,'CYCLE BOND'
 ! map the bonds in molecule
 allocate(bond_2_mol(N_TYPE_MOLECULES),bond_2_mol_end(N_TYPE_MOLECULES))
 bond_2_mol=-999 ; bond_2_mol_end=-999
 do i = 1, N_TYPE_MOLECULES
 iii=0; iii1=0; iii2=0
 do j = 1, how_many
   l_1=where_bond_starts(j)%line>where_mol_starts(i)%line.and.where_bond_starts(j)%line<where_mol_ends(i)%line
   l_2=where_bond_ends(j)%line>where_mol_starts(i)%line.and.where_bond_ends(j)%line<where_mol_ends(i)%line
   if (l_1) bond_2_mol(i) = j
   if (l_2) bond_2_mol_end(i) = j

   if (l_1) then
     iii1=iii1+1
     if (iii1>1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of BOND_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence BOND_TYPE '
                STOP
     endif
   endif

   if (l_2) then
      iii2=iii2+1
      if (iii2>1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of END_BOND_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence END_BOND_TYPE '
                STOP
     endif
   endif

   if (l_1.and.l_2) then
      iii = iii + 1
   endif
   if (iii > 1) then
        PRINT*, 'ERROR: in file:',trim(nf),' ONLY 1 definition of BOND_TYPE is allowed per MOLECULE_TYPE..END_MOLECULE_TYPE',&
                'See input file between the lines:',where_mol_starts(i)%line,where_mol_ends(i)%line,&
                ' and delete a sequence BOND_TYPE ... END_BOND_TYPE'
                STOP
   endif
  enddo
  enddo


  do j = 1, how_many
  l_1 = .false.
  l_2 = .false.
  do i = 1, N_TYPE_MOLECULES
   l_1=l_1.or.(where_bond_starts(j)%line>where_mol_starts(i)%line.and.where_bond_starts(j)%line<where_mol_ends(i)%line)
   l_2=l_2.or.(where_bond_ends(j)%line>where_mol_starts(i)%line.and.where_bond_ends(j)%line<where_mol_ends(i)%line)
  enddo
  if (.not.l_1) then
     PRINT*, 'ERROR: in file:',trim(nf),' BOND_TYPES oustide MOL...ENDMOL; delete the see line:', where_bond_starts(j)%line
     STOP
  endif
  if (.not.l_1) then
     PRINT*, 'ERROR: in file:',trim(nf),' BOND_TYPES oustide MOL...ENDMOL; delete the see line:', where_bond_ends(j)%line
     STOP
  endif
  enddo

 do i = 1, N_TYPE_MOLECULES
  if (bond_2_mol(i) > 0 .and. bond_2_mol_end(i) < 0) then
     print*, 'ERROR: in file ',trim(nf), 'BOND_TYPE defined but END_BOND_TYPES not defined ',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and bond def : see at line' ,where_bond_starts(bond_2_mol(i))%line
     STOP
  endif
  if (bond_2_mol(i) < 0 .and. bond_2_mol_end(i) > 0) then
     print*, 'ERROR: in file ',trim(nf), 'END_BOND_TYPE defined but BOND_TYPES not defined ',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and bond def : see at line' ,where_bond_ends(bond_2_mol_end(i))%line
     STOP
  endif

  if (bond_2_mol(i) /= bond_2_mol_end(i)) then
    if (bond_2_mol(i)>0.and.bond_2_mol_end(i)>0) then
     print*, 'ERROR: BOND ... END BOND is not within the same MOL .. END MOL; see in infile mol def between lines',&
     where_mol_starts(i)%line,where_mol_ends(i)%line, ' and bond def :' ,&
     where_bond_starts(bond_2_mol(i))%line,where_bond_ends(bond_2_mol_end(i))%line
     STOP
    else
     print*, 'ERROR: BOND definition outside any MOL definition; delete bond ... endbond that is outside mol..end_mol'
    endif
  endif
 enddo

 iii1=0;iii2=0
 do i = 1, N_TYPE_MOLECULES
   if (bond_2_mol(i)>0) iii1 = iii1+1
   if (bond_2_mol_end(i)>0) iii2 = iii2+1
 enddo

 if (iii1 /= how_many) then
    print*, 'ERROR in file ', trim(nf),' there are BOND_TYPE records outside MOL...END_MOL definition. Delete them.'
    STOP
 endif
 if (iii2 /= how_many) then
    print*, 'ERROR in file ', trim(nf),' there are END_BOND_TYPE records outside MOL...END_MOL definition. Delete them.'
    STOP
 endif


 do i = 1, N_TYPE_MOLECULES
   if (bond_2_mol(i) > 0) then
   j = bond_2_mol(i)
   iline = where_bond_starts(j)%line
   if (NumberOfWords(iline)<2) then
      print*, 'ERROR: More records needed at line ',iline
      STOP
   endif
   NNN = the_words(iline,2)%length
   call attempt_integer_strict(NNN,the_words(iline,2)%ch(1:NNN), &
                                        Nm_type_bonds(i)  , trim(nf), iline)
   do iline = where_bond_starts(j)%line+1, where_bond_ends(j)%line-1     
     l_id = .false.
     do iii1 = 4 , NumberOfWords(iline)
          NNN = the_words(iline,iii1)%length
           do k = 1, NNN
             chtemp(k:k) = UP_CASE(the_words(iline,iii1)%ch(k:k))
           enddo
        if (chtemp(1:NNN) == '*CONS') then
!           Nm_type_constrains(i) = Nm_type_constrains(i) + 1
           l_id = .true.
        endif 
     enddo      
     NNN = the_words(iline,4)%length
     call select_int_real_text_type(NNN,the_words(iline,4)%ch(1:NNN), i_type)
     if (i_type==1) then ! integer or text
       call attempt_integer_strict(NNN,the_words(iline,4)%ch(1:NNN), &
                                        i_index  , trim(nf), iline)
       if (N_predef_ff_bonds > 0) then
       if (i_index > lbound(predef_ff_bond,dim=1)-1 .and. i_index < ubound(predef_ff_bond,dim=1)+1) then
         if (predef_ff_bond(i_index)%is_constrain.or.l_ID) then 
           Nm_type_constrains(i) = Nm_type_constrains(i) + 1
         endif
       else
        print*, 'ERROR in file ',trim(nf), ' the record from line ',iline, ' coloumn 4 must be in the range', &
        lbound(predef_ff_bond,dim=1),ubound(predef_ff_bond,dim=1), ' instead of its value :',i_index
        STOP
       endif
       endif 
     endif  !  (i_type==1)
     if (i_type==3) then 
        do iii2 = 1, N_predef_ff_bonds
          if (trim(predef_ff_bond(iii2)%name)==trim(the_words(iline,4)%ch(1:NNN))) then
          if (predef_ff_bond(iii2)%is_constrain.or.l_ID) then
                       Nm_type_constrains(i) = Nm_type_constrains(i) + 1
          endif
          endif ! 
        enddo  ! do iii2 = 1, N_predef_ff_bonds
     endif    !if (i_type==3) then
   enddo
   endif
 enddo
ENDIF ! more than 0 bonds

print*, 'Nm_type_bonds=',Nm_type_bonds
print*,'Nm_type_constrains=',Nm_type_constrains
