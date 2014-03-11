module DoDummyInit_module
implicit none
public :: Do_Dummy_Init

contains
     Subroutine Do_Dummy_Init
     use ALL_dummies_data
     use ALL_atoms_data, only : Natoms, is_dummy,i_type_atom,is_sfield_constrained
     use atom_type_data, only : atom_type_name
     use connectivity_ALL_data, only : list_bonds,list_angles, Nbonds,Nangles,&
         list_14,size_list_14,list_excluded,size_list_excluded,list_excluded_HALF,size_list_excluded_HALF,&
         is_bond_dummy, &
         list_excluded_HALF_no_SFC, size_list_excluded_HALF_no_SFC,list_excluded_sfc_iANDj_HALF, &
         size_list_excluded_sfc_iANDj_HALF
     use sizes_data, only : N_pairs_sfc_sfc_123, N_pairs_14


      implicit none
      integer ibond,ibend,ineigh,iox,iex,jtmp,j,kk,ipos, iStyle,jStyle,k,k1
      integer i,idat,iat,jat,itype,jtype,iatConnect
      integer idmy,jdmy
      real(8) cc, bb, aa
      logical l_1
      logical, allocatable :: HasBeenChanged(:)
      integer  idummy, jdummy,  jdat, itmp
      logical HasBeenFound

      integer, allocatable :: temp_list_14(:,:), temp_size_list_14(:), next_site(:)

  allocate(next_site(Natoms))
  next_site = 0


    do idmy = 1, Ndummies
       i = map_dummy_to_atom(idmy)
       l_1 = .false.
       do j=1,Nbonds
           if (list_bonds(1,j)==i) then
             iat=list_bonds(2,j)
             l_1=.true.
           end if
           if (list_bonds(2,j)==i) then
             iat=list_bonds(1,j)
             l_1=.true.
           end if
        end do
        if (.not.l_1) then   ! DISCONECTED CASE 
           print*, ' ERROR: in DoDummyInit the dummy atom ', i, ' ', trim(atom_type_name(i_type_atom(i))), &
           ' is not conected to any bond. It cannot therefore be treated by the DoDummy procedure'
           STOP
        endif       
        iStyle = i_Style_dummy(idmy)
        select case (iStyle)
        case(1,2)
            do j=1,Nangles
              if ((list_angles(2,j)==iat).and.(list_angles(1,j)/=i).and.(list_angles(3,j)/=i)) then
                 all_dummy_connect_info(idmy,1)=list_angles(1,j)
                 all_dummy_connect_info(idmy,2)=list_angles(2,j)
                 all_dummy_connect_info(idmy,3)=list_angles(3,j)
               endif
            end do      ! plus-minus (up-dow
         case (3)
            do j=1,Nbonds
             if (list_bonds(1,j)==iat.and.(.not.is_dummy(list_bonds(2,j))))then
               iatConnect=list_bonds(2,j)
               l_1=.true.
             end if
             if (list_bonds(2,j)==iat.and.(.not.is_dummy(list_bonds(1,j))))then
               iatConnect=list_bonds(1,j)
               l_1=.true.
             end if
            end do
                 if (.not.l_1) then   ! DISCONECTED CASE 
                 print*, ' ERROR: in DoDummyInit: 3: the dummy atom ', i, ' ', trim(atom_type_name(i_type_atom(i))), &
                 ' is not conected via any bond. It cannot therefore be treated by the DoDummy procedure'
                 STOP
                 endif  
             do j=1,Nangles
              if (list_angles(2,j)==iatConnect.and.list_angles(1,j)/=iat.and.list_angles(3,j)/=iat) then
                 all_dummy_connect_info(idmy,1)=list_angles(1,j)
                 all_dummy_connect_info(idmy,2)=iat
                 all_dummy_connect_info(idmy,3)=list_angles(3,j)
              endif
            end do      ! plus-minus (up-down) of Lp in a pair will be assigned later                 
          case default
            print*, "LpGeomType should be 1,2,3 for this version; Actually case 1 not implemented"
            print*, "found LpGeomType",iStyle
            STOP
          end select      

    enddo  !idmy = 1, Ndummies

!C     for all Lp atoms
!C     1. Copy oxygen list_14()
!C     2. find the connected oxygen and include in the list_excluded_HALF()
!C     for the Lp atom all atoms from the oxygen list_excluded_HALF() except itself
!C

    do idmy = 1, Ndummies
       i = map_dummy_to_atom(idmy)
       iox=all_dummy_connect_info(idmy,2)
       size_list_14(i) = size_list_14(iox)
       list_14(i,1:size_list_14(i))=list_14(iox,1:size_list_14(i))
       size_list_excluded_HALF(i) = size_list_excluded_HALF(iox) - 1
       j=0
       do k = 1, size_list_excluded_HALF(iox)
           if (list_excluded_HALF(iox,k) /= i) then
               j=j+1
               list_excluded_HALF(i,j)=list_excluded_HALF(iox,k)
           endif
       enddo
       do iat=1,iox
          do k=1,size_list_14(iat)
            if (list_14(iat,k)==iox) then
              kk=size_list_14(i)+1
              size_list_14(i)=kk
              list_14(i,kk)=iat
            end if
          end do
        end do
        do iat=1,iox
          do k=1,size_list_excluded_HALF(iat)
            if (list_excluded_HALF(iat,k)==iox) then
              kk=size_list_excluded_HALF(i)+1
              size_list_excluded_HALF(i)=kk
              list_excluded_HALF(i,kk)=iat
            end if
          end do
        end do       
       
      end do ! idmy
!      
!      
!


      do idmy=1,Ndummies
        i=map_dummy_to_atom(idmy)
        j=0
        do k=1,size_list_excluded_HALF(i)
          iex=list_excluded_HALF(i,k)
          if (iex < i) then
            kk=size_list_excluded_HALF(iex)+1
            size_list_excluded_HALF(iex)=kk
            list_excluded_HALF(iex,kk)=i
           else
            j=j+1
            list_excluded_HALF(i,j)=iex
          endif
        end do
        size_list_excluded_HALF(i)=j
      end do
!C
!C  *** do the same for the 1-4 exclude list
!C



        do idmy=1,Ndummies
        i=map_dummy_to_atom(idmy)
        j=0
        do k=1,size_list_14(i)
          iex=list_14(i,k)
          if (iex.lt.i) then
            kk=size_list_14(iex)+1
            size_list_14(iex)=kk
            list_14(iex,kk)=i
           else
            j=j+1
            list_14(i,j)=iex
          endif
        end do
        size_list_14(i)=j
      end do

!C
!C  *** remove duplicates from the excluded list of dummy atoms
!C
      do idmy=1,Ndummies
        i=map_dummy_to_atom(idmy)
        do k=1,size_list_excluded_HALF(i)-1
          iex=list_excluded_HALF(i,k)
          ipos=0
          do kk=k+1,size_list_excluded_HALF(i)
            if (list_excluded_HALF(i,kk)==iex) ipos=kk !duplicate is located at ipos
          end do
          if (ipos > 1) then ! remove the atom at ipos
            size_list_excluded_HALF(i)=size_list_excluded_HALF(i)-1
            do kk=ipos,size_list_excluded_HALF(i)
              list_excluded_HALF(i,kk)=list_excluded_HALF(i,kk+1)
            end do
          end if
        end do
      end do

!C
!C  ***  add Lp-O-X-O-Lp to the excluded list
!C

      do idat=1,Ndummies
        idummy=map_dummy_to_atom(idat)
        iox=all_dummy_connect_info(idat,2)
        itmp=size_list_excluded_HALF(iox)
        do ineigh=1,itmp
          do iat=1,Ndummies
            jat=all_dummy_connect_info(iat,2)
            if (list_excluded_HALF(iox,ineigh).eq.jat) then
              jdummy=map_dummy_to_atom(iat)
!C
!C             *** add jdummy to idummy listex if it is not there
!C
              HasBeenFound=.false.
              do i=1,size_list_excluded_HALF(idummy)
                if (list_excluded_HALF(idummy,i).eq.jdummy) HasBeenFound=.true.
              end do
              if (.not.HasBeenFound) then
               if(idummy.lt.jdummy) then
                size_list_excluded_HALF(idummy)=size_list_excluded_HALF(idummy)+1
                list_excluded_HALF(idummy,size_list_excluded_HALF(idummy))=jdummy
               else
                size_list_excluded_HALF(jdummy)=size_list_excluded_HALF(jdummy)+1
                list_excluded_HALF(jdummy,size_list_excluded_HALF(jdummy))=idummy
               end if
!D              write(6,*) "idummy,jdummy",idummy,jdummy
              end if
            end if
          end do  ! next iat
        end do    ! next ineigh
      end do      ! next idat
!C
!C  ***  add Lp-O-X-X-O-Lp to the list14
!C


      do idat=1,Ndummies
        idummy=map_dummy_to_atom(idat)
        iox=all_dummy_connect_info(idat,2)
        itmp=size_list_14(iox)
        do ineigh=1,itmp
          do iat=1,Ndummies
            jat=all_dummy_connect_info(iat,2)
            if (list_14(iox,ineigh).eq.jat) then
              jdummy=map_dummy_to_atom(iat)
!C
!C             *** add jdummy to idummy list14 if it is not there
!C
              HasBeenFound=.false.
              do i=1,size_list_14(idummy)
                if (list_14(idummy,i).eq.jdummy) HasBeenFound=.true.
              end do
              if (.not.HasBeenFound) then
               if(idummy.lt.jdummy) then
                size_list_14(idummy)=size_list_14(idummy)+1
                list_14(idummy,size_list_14(idummy))=jdummy
               else
                size_list_14(jdummy)=size_list_14(jdummy)+1
                list_14(jdummy,size_list_14(jdummy))=idummy
               end if
!D              write(6,*) "list14:idummy,jdummy",idummy,jdummy
              end if
            end if
          end do  ! next iat
        end do    ! next ineigh
      end do      ! next idat
!
!
!



      allocate(HasBeenChanged(Ndummies)) ; HasBeenChanged(:)=.false.
      do idmy=1,Ndummies
       if (.not.HasBeenChanged(idmy)) then
        HasBeenChanged(idmy)=.true.
        i=map_dummy_to_atom(idmy)
        iStyle = i_style_dummy(idmy)
        i = all_dummy_connect_info(idmy,2)        
        cc = all_dummy_params(idmy,3)
        aa = all_dummy_params(idmy,1)
        select case (iStyle)
        case(1)
          if (dabs(cc) > 1.0d-6) then
             write(6,*) "WARNING (preparing dummies) C param should be zero for this DUMMY geometry type",iStyle, i
          end if
          do jdmy=idmy+1,Ndummies  ! check for mistakes
            j=all_dummy_connect_info(jdmy,2)
            jStyle = i_style_dummy(jdmy) ! a dummy atom type
            if (i==j) then     ! two dummy force centers are connected to the same atom
              write(6,*) "WARNING: Two extended force centers have the same position", i,j, '> ', idmy,jdmy
            end if
          end do
        case(2,3)
          if (dabs(cc) < 1.0d-6) then
             write(6,*) "C param should be non-zero for this Lp type", iStyle, i, idmy
          end if
          if (dabs(aa) < 1.0d-6) then
             write(6,*) "A param should be non-zero for this Lp type", iStyle,i, idmy
          end if
          l_1 = .false.
          do jdmy=idmy+1,Ndummies  ! find a twin
            j=all_dummy_connect_info(jdmy,2)
            jStyle = i_style_dummy(jdmy) ! a dummy atom type

            if (i == j) then     ! two dummy force centers are connected to the same atom
               if (l_1) then
                  write(6,*)"Found more than one dummy center in a pair" ,i,jdmy
               end if
               l_1=.true.
               jtmp=jdmy
             endif
            enddo
            if (l_1 ) then
                 all_dummy_connect_info(jtmp,1)=all_dummy_connect_info(idmy,3)
                 all_dummy_connect_info(jtmp,3)=all_dummy_connect_info(idmy,1)
                 HasBeenChanged(jtmp)=.true.
            else
                 write(6,*) "Dummy",i,idmy,iStyle," might not have a pair"
            end if
       
        case default
          print*, 'ERROR in DoDummyIniti (after connectivity) not defined style of dummy', iStyle, ' for atom dmy ', i,idmy
        end select
       end if     !  (.not.HasBeenChanged(i))
      end do

      do i = 1, Nbonds
      if (is_dummy(list_bonds(1,i)).or.is_dummy(list_bonds(2,i))) then
         is_bond_dummy(i) = .true.
      else
         is_bond_dummy(i) = .false.
      endif
      enddo

      deallocate(HasBeenChanged)


 list_excluded=0
 size_list_excluded=0
 include 'finalize_list_exclude.frg'

      end subroutine Do_Dummy_Init

end module DoDummyInit_module
