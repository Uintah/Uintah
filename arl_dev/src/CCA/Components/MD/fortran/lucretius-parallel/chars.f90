module chars

 public :: my_char_to_int
 public :: my_char1
 public :: char_intN_ch
 public :: char_intN_ch_NOBLANK
 public :: get_real_from_string
 public :: l_string_is_number
 public :: get_integer_from_string
 public :: UP_CASE
 public :: w_WP_CASE
 public :: search_one_word_in_line
 public :: l_eq_ch
 public :: scan_size_of_file
 public :: get_number_of_words
 public :: get_the_words
 public :: get_text_from_file
 public :: get_words_from_file
 public :: locate_word_in_key
 public :: locate_UPCASE_word_in_key
 public :: search_words
 public :: search_words_gi
 public :: select_int_real_text_type
 public :: get_integer_after_word
 public :: get_real_after_word
 public :: get_text_after_word
 public :: search_file_for_starting_word
 public :: attempt_integer_strict
 public :: attempt_real_strict
 public :: decrease_index1_by_1
 public :: get_index1_from_integer

contains
  subroutine my_char_to_int(i,ch)
 character(len=1) , intent (IN) :: ch
 integer , intent(OUT) :: i
 select case(ch)
  case('1')
   i=1
  case('2')
   i=2
  case('3')
   i=3
  case('4')
   i=4
  case('5')
   i=5
  case('6')
   i=6
  case('7')
   i=7
  case('8')
   i=8
  case('9')
   i=9
  case('0')
   i=0
  end select
  end subroutine my_char_to_int

 subroutine  my_char1(i,ch)
 character(len=1),intent(OUT):: ch
 integer,intent(IN):: i
 select case(i)
  case(1)
   ch='1'
  case(2)
   ch='2'
  case(3)
   ch='3'
  case(4)
   ch='4'
  case(5)
   ch='5'
  case(6)
   ch='6'
  case(7)
   ch='7'
  case(8)
   ch='8'
  case(9)
   ch='9'
  case(0)
   ch='0'
  end select
 end subroutine  my_char1

 subroutine char_intN_ch(N,Nr,the_string)
 integer, intent(IN) :: N,Nr
 character(len=N),intent(OUT):: the_string
 character(len=1) ch
 integer iv(N),Nr1,i

 Nr1=Nr
 do i=1,N
  iv(i)=Nr1/10**(N-i)
  Nr1=Nr1-iv(i)*10**(N-i)
  call my_char1(iv(i),ch)
  the_string(i:i)= ch !my_char1(i)
 enddo

 end subroutine char_intN_ch

  subroutine char_intN_ch_NOBLANK(N,Nr,the_string)
 integer, intent(IN) :: N,Nr
 character(len=N),intent(OUT):: the_string
 character(len=1) ch
 integer iv(N),Nr1,i,i1,i_first

 if (Nr == 0) then
   the_string='0'
   RETURN
 endif

 i1  =  0 
 Nr1=Nr
 do i=1,N
  iv(i)=Nr1/10**(N-i)
  Nr1=Nr1-iv(i)*10**(N-i)
  call my_char1(iv(i),ch)
  the_string(i:i)= ch !my_char1(i)
 enddo

 i1=0
 do i = 1,N
  if (the_string(i:i) /= '0') then
   i_first = i
   goto 3
  endif
 enddo

 3 continue 
 
 do i = i_first, N
  the_string(i-i_first+1:i-i_first+1) = the_string(i:i)
  the_string(i:i) = ' '
 enddo

 end subroutine char_intN_ch_NOBLANK


 subroutine get_real_from_string(size_string,the_string, real_number)
 implicit none
 real(8), intent(OUT) :: real_number
 integer , intent(IN) :: size_string
 character(1) , intent(IN) :: the_string(size_string)
 logical l1,l_string_is_number
 integer k,len,isuma, i,i_sign, N1,N2,kkk1,kkk2,kkk3 , kkk2_1,  kkk2_2, kkk1_1, kkk1_2, i_start_with
 integer, allocatable :: i1(:)
  logical l_decimal_point,l_mantisa
  integer point_is_at, mantisa_is_at
   real(8) rp_kkk2, rpp1
!  character(1) UP_CASE

!     print*, ' in get_real=',the_string(1:size_string)
  allocate(i1(size_string))
  i_sign=1
  if (the_string(1).eq.'-') i_sign=-1
  point_is_at=0
  l_decimal_point=.false. ; l_mantisa=.false.
  point_is_at=size_string+1
  mantisa_is_at=size_string+1
  do i=1,size_string
   if (the_string(i).eq.'.') then
      l_decimal_point=.true.
      point_is_at=i
   endif
   if (the_string(i).eq.'D'.or.the_string(i).eq.'E'.or.the_string(i).eq.'d'.or.the_string(i).eq.'e') then
     l_mantisa=.true.
      mantisa_is_at=i
   endif
   enddo
!   print*, 'point is at mantisa t : ', point_is_at, mantisa_is_at
   if (the_string(1).ne.'+'.and. the_string(1).ne.'-') then
     i_start_with=1
   else
     i_start_with=2
   endif
   if (point_is_at.gt.8) then
     call get_integer_from_string(8, the_string(i_start_with:i_start_with+8), kkk1_1 )
     call get_integer_from_string(point_is_at-i_start_with-8, the_string(8+i_start_with: point_is_at-i_start_with), kkk1_2)
     rpp1=dble(kkk1_1)*10**(point_is_at-i_start_with-8)
!     print*, 'rpp1=',rpp1
     rpp1=rpp1+dble(kkk1_2)
!     print*, 'rpp1=',rpp1
    else
     call get_integer_from_string(point_is_at-1, the_string(1:point_is_at-1),kkk1)
     rpp1=dble(kkk1)
!print*, 'rpp1=',rpp1
   endif
   N1=mantisa_is_at-1-(point_is_at+1)+1
!print*, 'N1=',N1
   if (N1.gt.8) then
    call get_integer_from_string(8,the_string(point_is_at+1:point_is_at+1+8-1),kkk2_1)
    call get_integer_from_string(N1-8, the_string(point_is_at+1+8 : mantisa_is_at-1), kkk2_2 )
    rp_kkk2=dble(kkk2_1)*10**(mantisa_is_at-1-(point_is_at+1+8-1) )
!    print*, kkk2_1 , kkk2_2 , 'kkks'
!    print*, 'rp_kkk2=',rp_kkk2
    rp_kkk2=rp_kkk2+dble( kkk2_2)
!    print*, 'rp_kkk2=',rp_kkk2
   else
    call get_integer_from_string(N1,the_string(point_is_at+1:mantisa_is_at-1),kkk2)
    rp_kkk2=dble(kkk2)
!print*, 'rp_kkk2=',rp_kkk2
   endif

   N2=size_string-(mantisa_is_at+1)+1
!print*, 'N2=',N2 , mantisa_is_at+1, size_string
   if (N2.gt.0) then
   call get_integer_from_string(N2,the_string(mantisa_is_at+1:size_string),kkk3)
   real_number=(dabs(rpp1)+rp_kkk2/10.0d0**dble(N1) ) *10**dble(kkk3)*dble(i_sign)
   else
   real_number=(dabs(rpp1)+rp_kkk2/10.0d0**dble(N1) )  *dble(i_sign)
   endif
!   print*, 'in get_real real_nrt=',real_number, kkk1,kkk2,kkk3
 deallocate(i1)
 end subroutine get_real_from_string


 logical function l_string_is_number(ch)
 character(1) , intent(IN) :: ch
 l_string_is_number = (ch == '1').or.(ch == '2').or.(ch == '3').or.(ch=='4').or.(ch=='5').or.(ch=='6') &
                                         .or.(ch ==  '7').or.(ch=='8').or.(ch=='9').or.(ch=='0')
 end  function l_string_is_number


 subroutine get_integer_from_string(size_string,the_string, KKK)
 implicit none
 integer, intent(OUT) :: KKK
 integer , intent(IN) :: size_string
 character(1) , intent(IN) :: the_string(size_string)
 logical l1!,l_string_is_number
 integer k,len,isuma, i, i_sign, k3
 integer, allocatable :: i1(:)
 allocate(i1(size_string))

 len=0
 k=0
 k3=0
 l1=.true.
i_sign=1
 do while(l1.and.k.lt.size_string)
   k=k+1
   k3=k3+1
   l1=l_string_is_number(the_string(k))
   if (k.eq.1) then
     if (the_string(k)=='-' ) i_sign=-1
     l1=l1.or.the_string(k)=='-'.or.the_string(k)=='+'
     if( the_string(k)=='-'.or.the_string(k)=='+') then
        k3=k3-1
      endif
    endif

     if (l_string_is_number(the_string(k))) then
       call my_char_to_int(i1(k3),the_string(k))
       len=len+1
     endif
 enddo

 if (len == 0) then
    KKK=0
 else
     isuma=0
    do i=1,len
       isuma=isuma+10**(len-i)*i1(i)
    enddo
    KKK=isuma*i_sign
 endif
 deallocate(i1)

 end subroutine get_integer_from_string

  character(1) function UP_CASE(ch)
  implicit none
     character(1), intent(IN) :: ch
     UP_CASE=ch
     select case (ch)
      case ('a') ; UP_CASE='A'
      case ('b') ; UP_CASE='B'
      case ('c') ; UP_CASE='C'
      case ('d') ; UP_CASE='D'
      case ('e') ; UP_CASE='E'
      case ('f') ; UP_CASE='F'
      case ('g') ; UP_CASE='G'
      case ('h') ; UP_CASE='H'
      case ('i') ; UP_CASE='I'
      case ('j') ; UP_CASE='J'
      case ('k') ; UP_CASE='K'
      case ('l') ; UP_CASE='L'
      case ('m') ; UP_CASE='M'
      case ('n') ; UP_CASE='N'
      case ('o') ; UP_CASE='O'
      case ('p') ; UP_CASE='P'

      case ('r') ; UP_CASE='R'
      case ('s') ; UP_CASE='S'
      case ('t') ; UP_CASE='T'
      case ('u') ; UP_CASE='U'
      case ('v') ; UP_CASE='V'
      case ('x') ; UP_CASE='X'
      case ('y') ; UP_CASE='Y'
      case ('z') ; UP_CASE='Z'
      case ('q') ; UP_CASE='Q'
      case ('w') ; UP_case='W'
     end select
    end function UP_CASE

  subroutine w_WP_CASE(w)
    character(*), intent(INOUT) :: w
    integer i,N
    N = len(w)
    do i = 1, N
        w(i:i) = UP_CASE(w(i:i))
    enddo
  end subroutine w_WP_CASE

  subroutine search_one_word_in_line(words, word,results)
  use types_module, only : word_type,one_I_one_L_type
  implicit none
  character(*), intent(IN) :: word
  type(word_type),intent(IN) :: words(:)
  type(one_I_one_L_type) results
  integer nu, nl, i, j, NNN
  character(1), allocatable :: tword(:), tword1(:)
   results%l = .false.
   results%i = -1
   nu = ubound(words,dim=1)
   if (nu==0) return
   nl = lbound(words,dim=1)
   if (nl /= 1.or. nu<nl) then
     print*, 'ERROR in subroutine search_one_word_in_line nu < nl or nl /=1',nl,nu
     STOP
   endif
!print*, 'in search_one_word_in_line nu nl =', nl,nu
   do i = nl, nu
   if (len(word)== words(i)%length) then
   NNN = words(i)%length
    allocate(tword(1:NNN),tword1(1:NNN))
    do j = 1, words(i)%length
      tword(j) = up_case(words(i)%ch(j:j))
      tword1(j) = up_case(word(j:j))
    enddo
    if (l_eq_ch(tword(1:NNN)(1:1),tword1(1:NNN)(1:1))) then
     results%l = .true.
     results%i = i
     RETURN
    endif
   deallocate(tword,tword1)
   endif
   enddo

  end subroutine search_one_word_in_line

  logical function l_eq_ch(s1,s2)
  implicit none
  character(1), intent(IN) :: s1(:),s2(:)
  integer nu1, nl1, nu2, nl2,i
  nu1 = ubound(s1,dim=1)
  nl1 = lbound(s1,dim=1)
  nu2 = ubound(s2,dim=1)
  nl2 = lbound(s2,dim=1)
  if (nu1-nl1 /= nu2-nl2) then
    l_eq_ch = .false.
    RETURN
  endif

  do i = nl1,nu1
   if (s1(i)(1:1) /= s2(i)(1:1)) then
     l_eq_ch = .false.
     RETURN
   endif
  enddo
   l_eq_ch = .true.
   end function l_eq_ch

  subroutine scan_size_of_file(nf,lines, MaxCol) ! no of lines
  implicit none
  character(*) , intent(IN) :: nf
  integer , intent(OUT) :: lines, MaxCol
  logical l_not_EOF,end_of_line
  character(1) ch
  integer, allocatable :: itemp(:),itemp1(:)
  integer SizeOfTemp
  integer, parameter :: chunck = 100
  integer k1

  SizeOfTemp = chunck
  allocate(itemp(SizeOfTemp))
   l_not_EOF=.true.
   end_of_line=.false.
   lines = 0
   MaxCol=0
  open(unit=1131,file=trim(nf),status='old')
   do while (l_not_EOF)
   k1 = 0
   do while (.not.end_of_line)
      read(1131,'(A)',advance='NO',EOR=2,END=3) ch
      if (ch.eq.';') then
      endif
       k1 = k1 + 1
   enddo
2  continue
   lines = lines + 1
   if (lines > SizeOfTemp) then
      allocate(itemp1(SizeOfTemp+chunck))
      itemp1(1:SizeOfTemp) = itemp(1:SizeOfTemp)
      itemp1(lines) = k1
      deallocate(itemp)
      allocate(itemp(SizeOfTemp+chunck))
      itemp(1:lines) = itemp1(1:lines)
      deallocate(itemp1)
      SizeofTemp = SizeOfTemp+chunck
   else
      itemp(lines) = k1
   endif

   enddo
3 continue
!print*, 'Closing file'
  if (lines ==0) then
   close(1131)
   return
  endif

  close(1131)
  MaxCol = maxval(itemp(1:lines))

  end subroutine scan_size_of_file

  subroutine get_number_of_words(N, line, words)
  implicit none
  integer, intent(IN) :: N
  character(1), intent(in):: line(N)
  integer i,nl,nu
  character(1) ch_prev
  integer, intent(out) :: words

  nl = 1   !lbound(line,dim=1)
  nu = N    !ubound(line,dim=1)
  words = 0
  if ( nu==0) return

  ch_prev = line(nl)
  if (ch_prev /= ' ') words=1
  do i = nl+1,nu
    if (ch_prev == ' ') then
    if (line(i) /= ' ') then
      words = words + 1
    endif
    else
    endif
    ch_prev = line(i)
  enddo

 end subroutine get_number_of_words

  subroutine get_the_words(N, line, words, the_word)
  use types_module, only : word_type
  implicit none
  integer, intent(IN) ::N
  character(1), intent(in):: line(N)
  integer i,nl,nu,k,iw
  character(1) ch_prev
  integer, intent(in) :: words
  type(word_type) the_word(words)

  nl = 1   !lbound(line,dim=1)
  nu = N    !ubound(line,dim=1)
  iw = 0
  if ( nu==0) return
  the_word%length=0
  the_word%ch = ' '
  ch_prev = line(nl)
  if (ch_prev /= ' ') then
    iw=1
    k=1
    the_word(iw)%ch(k:k) = line(1)
    the_word(iw)%length  = 1
  endif

  do i = nl+1,nu
    if (ch_prev == ' ') then
          if (line(i) /= ' ') then
           k=1
           iw = iw + 1
           the_word(iw)%ch(k:k) = line(i)
           the_word(iw)%length = the_word(iw)%length+1
       endif
        else
          if (line(i) /= ' ') then
           k=k+1
           the_word(iw)%ch(k:k) = line(i)
           the_word(iw)%length = the_word(iw)%length+1
          endif
        endif
        ch_prev = line(i)
  enddo

 end subroutine get_the_words

  subroutine get_text_from_file(nf,lines, MaxCol,the_lines,SizeOfLine)
  implicit none
  integer, intent(IN) :: lines, MaxCol
  character(*), intent(IN) :: nf
  character(1), intent(OUT):: the_lines(lines,MaxCol)
  integer, intent(OUT) :: SizeOfLine(lines)
  logical end_of_line
  integer k1,i,j,k
  character(1) ch
 ! lines and MaxCol  must came as correct and previus determined by scan_size_of_file
  open(unit=1131, file=trim(nf))
  do i = 1, lines
!  print*, i
     end_of_line=.false.
         k1 = 0
         do while (.not.end_of_line)
      read(1131,'(A)',advance='NO',EOR=2,END=3) ch
      if (ch.eq.';') then
      endif
          k1 = k1 + 1
          the_lines(i,k1) = ch
   enddo
2  continue
   SizeOfLine(i) = k1

  enddo
  3 continue
!  print*, 'closing file 1131 in get text from file'
  close(1131)
 end  subroutine get_text_from_file



subroutine get_words_from_file(nf,lines, Max_words_per_line, NumberOfWords,the_words)
use types_module, only : word_type
implicit none
character(*), intent(IN):: nf
integer, intent(OUT) :: lines, Max_words_per_line
integer, allocatable, intent(OUT) :: NumberOfWords(:)
type(word_type), allocatable, intent(OUT) :: the_words(:,:)
character(1), allocatable :: the_lines(:,:)
integer, allocatable :: SizeOfLine(:)
integer i,j,k,col

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
!  allocate(l_skip_line(lines)); l_skip_line=.false.;
!  do i = 1, lines ;
!   if (SizeOfLine(i)==0.or.NumberOfWords(i)==0.or.the_lines(i,1)==comment_character_1.or.the_lines(i,1)==comment_character_2) then
!    l_skip_line(i) = .true.
!   endif
!  enddo
  
  deallocate(the_lines, SizeOfLine)
  
end subroutine get_words_from_file

 subroutine locate_word_in_key(key, N_valid_recs, the_words,l_found,i_position)
 use types_module, only : word_type,two_I_one_L_type,one_I_one_L_type
 implicit none
 character(*), intent(IN) :: key
 integer, intent(IN) :: N_valid_recs ! search for N_valid_recs more records after the key
 type(word_type), intent(IN):: the_words(:)
 logical, intent(OUT) :: l_found
 integer, intent(OUT) :: i_position
 
 integer i,istart,iend,NNN

 istart = lbound(the_words,dim=1)
 iend =ubound(the_words,dim=1)

 l_found = .false.
 i_position=0
 do i = istart  , iend
    NNN = the_words(i)%length
    if (trim(key) == trim(the_words(i)%ch(1:NNN))) then
      i_position = i
      l_found=.true.
      goto 3
    endif
 enddo
3 continue

 if (.not.l_found) RETURN
 
 if (i_position+N_valid_recs > iend) then
  print*, 'More records needed to define the key =',trim(key)
  STOP
 endif

 end subroutine locate_word_in_key

  subroutine locate_UPCASE_word_in_key(key, N_valid_recs, the_words,l_found,i_position)
 use types_module, only : word_type,two_I_one_L_type,one_I_one_L_type
 implicit none
 character(*), intent(IN) :: key
 integer, intent(IN) :: N_valid_recs ! search for N_valid_recs more records after the key
 type(word_type), intent(IN):: the_words(:)
 logical, intent(OUT) :: l_found
 integer, intent(OUT) :: i_position
 logical l_1
 integer i,istart,iend,NNN,k

 istart = lbound(the_words,dim=1)
 iend =ubound(the_words,dim=1)

 l_found = .false.
 i_position=0
 do i = istart  , iend
    NNN = the_words(i)%length
    l_1 = NNN == len(trim(key))
    if (l_1) then
    do k = 1,min(NNN,len(trim(key)))
      l_1 = l_1 .and. key(k:k) == UP_CASE(the_words(i)%ch(k:k))
    enddo
    endif
    if (l_1) then
      i_position = i
      l_found=.true.
      goto 3
    endif
 enddo
3 continue

 if (.not.l_found) RETURN

 if (i_position+N_valid_recs > iend) then
  print*, 'More records needed to define the key =',trim(key)
  STOP
 endif

 end subroutine locate_UPCASE_word_in_key

 
  

 subroutine search_words(istart,iend,lines,Max_words_per_line,the_words,SizeOfLine,NumberOfWords,word,&
                         l_skip_line,which,nf,lstrict,l_print_warning)
 use types_module, only : word_type,two_I_one_L_type,one_I_one_L_type
 implicit none
 logical, intent(IN) :: lstrict ! if lstrict ERROR else WARNING
 character(*), intent(IN) :: nf
 integer,intent(IN):: istart,iend
 integer, intent(IN)::lines,Max_words_per_line
 type(word_type), intent(IN) :: the_words(istart:iend,Max_words_per_line)  !(lines,Max_words_per_line)
 integer, intent(IN) :: SizeOfLine(istart:iend) !(lines)
 integer, intent(IN) :: NumberOfWords(istart:iend)   ! (lines)
 character(*), intent(IN) :: word
 logical,intent(inout) :: l_skip_line(istart:iend)  ! (lines)
 logical, intent(in) , optional :: l_print_warning
 type(two_I_one_L_type) which
 type(one_I_one_L_type) searched_word
 integer i,j,k
 logical l_do_print

  which%find=.false.
  which%line = 0
  which%word = 0

!  if (istart < 1 .or. iend > lines .or. iend < istart .or. (iend-istart+1) > lines) then
!   print* , 'ERROR in search_words in the file "',trim(nf),&
!            '"iend and istart and lines are not compatible; istart iend lines=',istart,iend, lines
!   print*,istart < 1 ,iend > lines , iend < istart,(iend-istart+1) > lines
!   STOP
!  endif
  do i = istart, iend ; 
  if (.not.l_skip_line(i)) then
!  print*, i,'<',iend,'go in with line=',SizeOfLine(i), NumberOfWords(i)
!read(*,*)
  call search_one_word_in_line(the_words(i,1:NumberOfWords(i)), word, searched_word)
!  print*, '------------------',searched_word%l,searched_word%i
  if(searched_word%l) then
     l_skip_line(i) = .true.
!     print*,  'MATCH IN LINE ', i, searched_word
      which%find=searched_word%l
      which%line = i
      which%word = searched_word%i
!print*, 'BINGO which =',which
      RETURN
   endif
  endif ; enddo

 if (.not.which%find) then
 if (lstrict) then
 print*, 'ERROR ; from the file "', trim(nf), '" Missing (or commented) keyword : ', trim(word), &
 ' see lines : ',istart,' to: ', iend
 STOP
 else
 l_do_print = .true.
  if (present(l_print_warning)) l_do_print = l_print_warning
  if (l_do_print)  &
   print*, 'WARNING ; from the file "', trim(nf), '" Missing (or commented) keyword : ', trim(word), &
   ' see lines : ',istart,' to: ', iend
 endif
 endif
!print*, 'exit search_words'
 end subroutine search_words

 subroutine search_words_gi(istart,iend,lines,nf,Max_words_per_line,the_words,SizeOfLine,NumberOfWords,&
                            word,l_skip_line,N,which,lstrict)
 use types_module, only : word_type,two_I_one_L_type,one_I_one_L_type
 implicit none
 logical, intent(IN) :: lstrict
 integer,intent(IN):: istart,iend
 integer, intent(IN)::lines,Max_words_per_line
 type(word_type), intent(IN) :: the_words(istart:iend,Max_words_per_line) !(lines,Max_words_per_line)
 character(*), intent(IN) :: nf
 integer, intent(IN) :: SizeOfLine(istart:iend) !(lines)
 integer, intent(IN) :: NumberOfWords(istart:iend)  !(lines)
 character(*), intent(IN) :: word
 logical,intent(inout) :: l_skip_line(istart:iend)  ! (lines)
 integer, intent(IN) :: N
 type(two_I_one_L_type) which(N)
 type(one_I_one_L_type) searched_word
 integer i,j,k,i1


  which%find=.false.
  which%line = 0
  which%word = 0
  i1 = 0

!  if (istart < 1 .or. iend > lines .or. iend < istart .or. (iend-istart+1) > lines) then
!   print* , 'ERROR in search_words in the file "',trim(nf),&
!            '"iend and istart and lines are not compatible; istart iend lines=',istart,iend, lines
!   STOP
!  endif
  do i = istart, iend ; if (.not.l_skip_line(i)) then
!  print*, 'go in with line=',the_lines(i,1:SizeOfLine(i))`
!print*, word,'searchfor:',searched_word,'"'
  call search_one_word_in_line(the_words(i,1:NumberOfWords(i)), word, searched_word)
  if(searched_word%l) then
     l_skip_line(i) = .true.
!     print*,  'MATCH IN LINE ', i, searched_word
      i1 = i1 + 1
      if (i1 > N) then
        write(6,*) ' In the file : "',trim(nf),&
        '" there are too many records "',trim(word), ' Clean the file  and restart the program'
        STOP
      endif
      which(i1)%find=searched_word%l
      which(i1)%line = i
      which(i1)%word = searched_word%i
   endif
  endif ; enddo


 end subroutine search_words_gi


  subroutine select_int_real_text_type(N,S,i_type)
  implicit none
  integer , intent(IN) :: N
  integer , intent(OUT) :: i_type
  character(1), intent(IN) :: S(N)
!  character(1) UP_CASE
!  logical l_string_is_number
  logical l1,l2
  integer k
  if (N.eq.0) then
     i_type=0
     RETURN
   endif

  l1= (S(1).eq.'+'.or.S(1).eq.'-'.or.S(1).eq.'.')
  l1=l1.or.l_string_is_number(S(1))
  if (.NOT.l1) then
     i_type=3 ! it's a text
     RETURN
  endif
  l1=.true.
  do k=2,N
   l2=(l_string_is_number(S(k)).or.S(k).eq.'+'.or.S(k).eq.'-'.or.S(k).eq.'.'.or.S(k).eq.'D'.or.S(k).eq.'E')
   l2=l2.or.S(k).eq.'d'.or.S(k).eq.'e'
  if (.not.l2) then
    i_type = 3 ! is  a text
    RETURN
  endif
 enddo

  do k=1,N
   if (S(k).eq.'D'.or.S(k).eq.'E'.or.S(k).eq.'d'.or.S(k).eq.'e'.or.S(k).eq.'.' ) then
     i_type=2
     RETURN
   endif
   enddo
   i_type=1   !it's a integer
  end subroutine select_int_real_text_type

 subroutine get_integer_after_word(nf,which,key,Nwords,my_words, the_result)
use types_module, only : two_I_one_L_type,word_type
implicit none
character(*), intent(IN) :: nf
character(*), intent(IN) :: key
type (two_I_one_L_type) which
integer, intent(IN) ::  Nwords
type(word_type) my_words(1:Nwords)
character(12) ch_asked_type
integer i,j,k, NNN, i_type
integer, intent(OUT) :: the_result

integer, parameter :: i_type_asked = 1

if (which%find) then
  if (which%word+1 > NWords ) then
   print*, 'ERROR in the file ',trim(nf), ' One more record is needed at the line ',which%line, 'after the record "',&
my_words(which%word)%ch(1:my_words(which%word)%length),'"'
   STOP
  endif
  NNN = my_words(which%word+1)%length
  call select_int_real_text_type(NNN,my_words(which%word+1)%ch(1:NNN),i_type)
  if (i_type.ne.i_type_asked ) then
   print*, 'ERROR in the file ',trim(nf), ' incompatible format at line ',which%line, ' instead of the record: ',&
    my_words(which%word+1)%ch(1:NNN), ' an integer is required'
   STOP
  endif
else
  write(6,*) 'ERROR in ',trim(nf), ' file : the keyword ', trim(key),' is not defined'
  STOP
endif

 call get_integer_from_string(NNN, my_words(which%word+1)%ch(1:NNN),the_result)

end subroutine get_integer_after_word

subroutine get_real_after_word(nf,which,key,Nwords,my_words, the_result)
use types_module, only : word_type,two_I_one_L_type
implicit none
character(*), intent(IN) :: nf
character(*), intent(IN) :: key
type (two_I_one_L_type) which
integer, intent(IN) ::  Nwords
type(word_type) my_words(1:Nwords)
character(12) ch_asked_type
integer i,j,k, NNN, i_type
real(8), intent(OUT) :: the_result

integer, parameter :: i_type_asked = 2

if (which%find) then
  if (which%word+1 > NWords ) then
   print*, 'ERROR in the file ',trim(nf), ' One more record is needed at the line ',which%line, 'after the record',&
my_words(which%word)%ch(1:my_words(which%word)%length)
   STOP
  endif
  NNN = my_words(which%word+1)%length
  call select_int_real_text_type(NNN,my_words(which%word+1)%ch(1:NNN),i_type)
  if (i_type.ne.i_type_asked ) then
   print*, 'ERROR in the file ',trim(nf), ' incompatible format at line ',which%line, ' instead of the record: ',&
    my_words(which%word+1)%ch(1:NNN), ' an integer is required'
   STOP
  endif
else
  write(6,*) 'ERROR in ',trim(nf), ' file : the keyword ', trim(key),' is not defined'
  STOP
endif

 call get_real_from_string(NNN, my_words(which%word+1)%ch(1:NNN),the_result)

end subroutine get_real_after_word

subroutine get_text_after_word(nf,which,key,Nwords,my_words, N, the_result)
use types_module,only: two_I_one_L_type,word_type
implicit none
character(*), intent(IN) :: nf
character(*), intent(IN) :: key
type (two_I_one_L_type) which
integer, intent(IN) ::  Nwords
type(word_type) my_words(1:Nwords)
character(12) ch_asked_type
integer i,j,k, NNN, i_type
integer, intent(IN) :: N
character(N), intent(OUT) :: the_result

integer, parameter :: i_type_asked = 3

if (which%find) then
  if (which%word+1 > NWords ) then
   print*, 'ERROR in the file ',trim(nf), ' One more record is needed at the line ',which%line, 'after the record',&
my_words(which%word)%ch(1:my_words(which%word)%length)
   STOP
  endif
  NNN = my_words(which%word+1)%length
  call select_int_real_text_type(NNN,my_words(which%word+1)%ch(1:NNN),i_type)
  if (i_type.ne.i_type_asked ) then
   print*, 'ERROR in the file ',trim(nf), ' incompatible format at line ',which%line, ' instead of the record: ',&
    my_words(which%word+1)%ch(1:NNN), ' an integer is required'
   STOP
  endif
else
  write(6,*) 'ERROR in ',trim(nf), ' file : the keyword ', trim(key),' is not defined'
  STOP
endif

 the_result = ' '
 do i = 1, min(N, NNN)
   the_result(i:i) =  my_words(which%word+1)%ch(i:i)
 enddo

end subroutine get_text_after_word

 subroutine search_file_for_starting_word(the_word,nf,i_index,l_found)
   character(*) , intent(IN) :: the_word
   character(*) , intent(IN) :: nf
   integer i_index
   logical l_found
   character(10000) ch
   character(1) ch1
   integer i,j,k,N,i0
   logical end_of_file, end_of_line

   open(unit=33,file=trim(nf),err=60)

   l_found = .false.
   i_index = 0
  
   end_of_file = .false.
   end_of_line = .false.

   N = len(the_word)
   do i = 1,10000 ; ch(i:i) = ' ' ; enddo ; 
   i0 = 0
   do while(.not.end_of_file) 
   i0 = i0 + 1
    k = 0
     do while (.not.end_of_line)
      
       read(33,'(A1)',advance='NO',END=4,EOR=3) ch1
       if (ch1.ne.' ') then
         k = k + 1
         ch(k:k) = ch1
       endif
       if (k.eq.N) then
         if (ch(1:N).eq.the_word(1:N) ) then
           i_index = i0
           l_found = .true.
           close(33)
           RETURN
         else
           read(33,'(A1)',advance='NO',END=4,EOR=3) ch1 ! do it so becouse I only want the very first word
           read(33,*)
           goto 3
        endif
       endif
      
     enddo ! while (.not.end_of_line
3 continue
   enddo !do while(.not.end_of_file)
 4 continue

   close(33)

   return

60  print*, 'file ',trim(nf) , 'does not exist'
  stop
 end subroutine search_file_for_starting_word 

 subroutine attempt_integer_strict(NNN,the_word, int_out, add_text, add_int)
! it will try to read an integer otherwise will stop the program
     implicit none
     integer, intent(IN) :: NNN, add_int
     integer, intent(OUT) :: int_out
     character(*) , intent(IN) :: the_word, add_text
     integer i_type
     call select_int_real_text_type(NNN,the_word(1:NNN),i_type)
     if (i_type.ne.1) then
         write(6,*) 'ERROR in file "',trim(add_text),'" at line ',add_int, &
         'an integer is required instead of record: ',the_word(1:NNN)
         STOP
     endif
     call get_integer_from_string(NNN, the_word(1:NNN), int_out  )
 end subroutine attempt_integer_strict

 subroutine attempt_real_strict(NNN,the_word, real_out, add_text, add_int)
! it will try to read an integer otherwise will stop the program
     implicit none
     integer, intent(IN) :: NNN, add_int
     real(8), intent(OUT) :: real_out
     character(*) , intent(IN) :: the_word, add_text
     integer i_type
     i_type = 1
     call select_int_real_text_type(NNN,the_word(1:NNN),i_type)
     if (i_type.ne.2) then
         write(6,*) 'ERROR in file "',trim(add_text),'" at line ',add_int, &
         'an real is required instead of record: ',the_word(1:NNN)
         STOP
     endif
     call get_real_from_string(NNN, the_word(1:NNN), real_out  )
 end subroutine attempt_real_strict

 subroutine decrease_index1_by_1(ch)
 character(*), intent(INOUT) :: ch
   integer locate(2)
   integer i,j,k,N,i1,i_type

 i1 = 0
 N=len(ch)
 do i = 1, N
 if (ch(i:i)=='_') then
  i1 = i1 + 1
  if (i1>2) then
     print*,'ERROR: in decrease_index1_by_1: too many (3) chars of "_" '
         STOP
  endif
  locate(i1) = i
 endif
 enddo
 locate(1) = locate(1) + 1
 locate(2) = locate(2) - 1
 N=locate(2)-locate(1)+1
 call select_int_real_text_type(N,ch(locate(1):locate(2)),i_type)
 if (i_type /= 1) then
      print*,'ERROR: in decrease_index1_by_1: NOT AN INTEGER LEFT '
         STOP
 endif

 call get_integer_from_string(N,ch(locate(1):locate(2)), KKK)
 KKK = KKK - 1

 call get_index1_from_integer(kkk,ch(1:N+2))


 end  subroutine decrease_index1_by_1

 subroutine get_index1_from_integer(kkk,ch)
 character(*) , intent(INOUT) :: ch
 integer , intent(IN) ::  kkk
 integer, parameter :: MXs=20
 integer i,j,k,i1,N
 logical :: done_first = .false.

 N=len(ch)
 ch(1:N) = ' '
 ch(1:1) = '_'
 i1 = 0
 do i = 1, MXs
   k = kkk / 10**(MXs-i)
   if (k /= 0.and..not.done_first) then
     i1 = i1 + 1
         done_first = i1>1
         call my_char1(k,ch(i1+1:i1+1) )
!        print*, i1, ' ch = ' , ch(1:i1+1)
   endif
 enddo
 ch(i1+1+1:i1+1+1) = '_'
 end  subroutine get_index1_from_integer



!!!!!!!!!!!!!!!!!!!




      character(1) function int2ch(i)
      implicit none
      integer i
      character(1) ch

      if(i==0) then
      ch='0'
      else if(i==1) then
      ch='1'
      else if(i==2) then
      ch='2'
      else if(i==3) then
      ch='3'
      else if(i==4) then
      ch='4'
      else if(i==5) then
      ch='5'
      else if(i==6) then
      ch='6'
      else if(i==7) then
      ch='7'
      else if(i==8) then
      ch='8'
      else if(i==9) then
      ch='9'
      else
      write(6,*)'ERROR in get_ch1_from_int: i not in the range 0 to 9'
      stop
      endif

      int2ch=ch

      end function int2ch


      character(4) function trail4(i)
      implicit none
      integer i
      integer im,imm
      character(4) ch
      if (i < 10) then
      ch(1:3) = '000'
      ch(4:4) = int2ch(i)
      else if (i>=10.and.i<100) then
      ch(1:2) = '00'
      ch(3:3) = int2ch(INT(i/10))
      ch(4:4) = int2ch(mod(i,10))
      else if (i>=100.and.i<1000) then
      ch(1:1) = '0'
      ch(2:2) = int2ch(INT(i/100))
      im = mod(i,100)
      ch(3:3) = int2ch(INT(im/10))
      ch(4:4) = int2ch(mod(im,10))
      else if (i>=1000.and.i<10000) then
      print*,i
      ch(1:1) = int2ch(INT(i/1000))
      im = mod(i,1000)
      ch(2:2) = int2ch(INT(im/100))
      imm = mod(im,100)
      ch(3:3) = int2ch(INT(imm/10))
      ch(4:4) = int2ch(mod(imm,10))
      else
      write(6,*) ' ERROR i out of range 0 .. 9999 in trail4';
      stop
      endif
      trail4 = ch

      end  function trail4


      character(3) function trail3(i)
      implicit none
      integer i
      integer im,imm
      character(3) ch
      if (i < 10) then
      ch(1:2) = '000'
      ch(3:3) = int2ch(i)
      else if (i>=10.and.i<100) then
      ch(1:1) = '0'
      ch(2:2) = int2ch(INT(i/10))
      ch(3:3) = int2ch(mod(i,10))
      else if (i>=100.and.i<1000) then
      ch(1:1) = int2ch(INT(i/100))
      im = mod(i,100)
      ch(2:2) = int2ch(INT(im/10))
      ch(3:3) = int2ch(mod(im,10))
      else
      write(6,*) ' ERROR i out of range 0 .. 999 in trail3';
      stop
      endif
      trail3 = ch

      end  function trail3



 
end module chars




 module digits_module
 implicit none
 public :: first_digit
 public :: first_and_second_digit
 contains
 integer function first_digit(a)
 real(8) a
 integer, parameter :: MX = 1000
 real(8) a1
 integer i,j,k,i1

 a1 = a

 if (a1 == 0.0d0) then
 first_digit = 0
 RETURN
 endif

 if (abs(a1) < 1.0d0) then
   do i = 1,MX
     if (int(abs(a1)) == 0) then
       a1=a1*10.0d0
     else   if (abs(a1)<10.0d0) then
       i1 = int(a1)
       goto 3
     endif
   enddo
 else if (abs(a1) > 1.0d0) then
    do i = 1, MX
      if (abs(a1) > 10.0d0) then
        a1 = a1 /10.0d0
      else
            i1  = int(a1)
        goto 3
      endif
    enddo
 else if (abs(a1) == 1.0d0) then
      i1 = 1
 endif

  3 continue

 if (dabs(a1) > 10.0d0) then
   print*, 'WEIRD ERROR in first_digit; |abs(a1)|>10',a1
   stop
 endif

 !print*, 'i1 =', i1,abs(i1)
 first_digit = i1
 RETURN

 end function first_digit

 subroutine first_and_second_digit(a,i1,i2,ipwd)
 real(8), intent(IN) :: a
 integer, intent(OUT) :: ipwd,i2,i1
 integer, parameter :: MX = 1000
 real(8) a1,a2,a3
 integer i,j,k,icy,i3

 a1 = a

 if (a1 == 0.0d0) then
 i1 = 0
 i2 = 0
 ipwd = 0
 RETURN
 endif

 icy = 0
 if (abs(a1) < 1.0d0) then
   do i = 1,MX
     if (int(abs(a1)) == 0) then
           icy = icy - 1
       a1=a1*10.0d0
     else   if (abs(a1)<10.0d0) then
       i1 = int(a1)
       goto 3
     endif
   enddo
 else if (abs(a1) > 1.0d0) then
    do i = 1, MX
      if (abs(a1) > 10.0d0) then
            icy = icy + 1
        a1 = a1 /10.0d0
      else
            i1  = int(a1)
        goto 3
      endif
    enddo
 else if (abs(a1) == 1.0d0) then
      i1 = 1
          i2 = 0
          ipwd = 0
          RETURN
 endif

  3 continue


 if (dabs(a1) > 10.0d0) then
   print*, 'WEIRD ERROR in first_and_second_digit; |abs(a1)|>10',a1
   stop
 endif

 ! get second digit

! print*,'a1=',a1
 if (icy > 1) then
  ipwd = icy
 else
  ipwd = icy
 endif

 a2 = abs( (a1 - dble(i1))*10.0d0  )
 i2 = INT(a2)
 a3 = abs( (a2-dble(i2))*10.0d0 )
 i3 = NINT(a3)
 if (i3 > 5)  i2 = i2 + 1
 if (i2 == 10) then
    i1 = i1 + 1
        i2 = 0
        if (i1 > 10) then
         if (a>0.0d0) then
            ipwd = ipwd + 1
         else
            ipwd = ipwd - 1
         endif
        endif
 endif


! print*,a,'in sub i1 i2 ipwd=',i1,i2,ipwd
! print*,'pwd=',10**(-2)
!
 end subroutine first_and_second_digit



 end module digits_module



