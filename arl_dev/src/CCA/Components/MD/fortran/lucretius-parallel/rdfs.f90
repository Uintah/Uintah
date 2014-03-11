! Does rdfs the statistics. 

MODULE rdfs_module


 implicit none

 private :: rdfs_get_instant
 private :: rdfs_print
 public :: drivers_rdfs_eval

 real(8),allocatable , private :: gr(:,:),gr_all_bins(:) ,gr_par(:,:),gr_all_bins_par(:),gr_perp(:,:),gr_all_bins_perp(:)
 real(8), allocatable, private :: RA_gr_par(:,:),RA_gr_perp(:,:),RA_gr(:,:)
 contains

   subroutine drivers_rdfs_eval
   use rdfs_data, only : rdfs
   use integrate_data, only : integration_step

   implicit none
   real(8),save :: di_counter_roll_avg_rdfs = 0.0d0
   logical,save :: l_very_first = .true.
   if (.not.rdfs%any_request) RETURN 

   if (l_very_first) then
     call very_first_pass
     l_very_first=.false.
   endif

   if (mod(integration_step,rdfs%N_collect) == 0) then
! Evaluation of it was done in short forces loops
     if (mod(integration_step,rdfs%N_print)==0) then
         di_counter_roll_avg_rdfs=di_counter_roll_avg_rdfs+1.0d0
         call rdfs_print( di_counter_roll_avg_rdfs )
     endif
   endif
  CONTAINS
  subroutine very_first_pass
  use rdfs_data, only : l_details_rdf_CTRL,rdfs,N_BIN_rdf
       if (rdfs%N_Z_BINS > 1) then
         allocate(gr(0:N_BIN_rdf,1:rdfs%N_pairs)); gr=0.0d0
         allocate(gr_all_bins(1:rdfs%N_pairs))   ; gr_all_bins=0.0d0
         allocate(RA_gr(0:N_BIN_rdf,1:rdfs%N_pairs)); RA_gr=0.0d0
         if (l_details_rdf_CTRL) then
           allocate(gr_par(0:N_BIN_rdf,1:rdfs%N_pairs)); gr_par=0.0d0
           allocate(gr_all_bins_par(1:rdfs%N_pairs))   ; gr_all_bins_par=0.0d0
           allocate(gr_perp(0:N_BIN_rdf,1:rdfs%N_pairs)); gr_perp=0.0d0
           allocate(gr_all_bins_perp(1:rdfs%N_pairs))   ; gr_all_bins_perp=0.0d0
           allocate(RA_gr_par(0:N_BIN_rdf,1:rdfs%N_pairs)); RA_gr_par=0.0d0
           allocate(RA_gr_perp(0:N_BIN_rdf,1:rdfs%N_pairs)); RA_gr_perp=0.0d0
         endif      
       endif
  end subroutine very_first_pass
  end subroutine drivers_rdfs_eval

  subroutine rdfs_print( di_counter )
  use rdfs_data, only : RA_gr_counters,RA_gr_counters_par,RA_gr_counters_perp,&
                        gr_counters,gr_counters_par,gr_counters_perp,&
                        l_details_rdf_CTRL,rdfs, BIN_rdf, N_BIN_rdf
   use file_names_data, only : continue_job, path_out
   use chars, only : char_intN_ch
   use sim_cel_data, only : sim_cel


   real(8), intent(IN) :: di_counter
   real(8) fct_prev, fct_next
   integer i,j,k,NB
   real(8) , allocatable :: Z_axis(:)
   character(4) ch4
   logical, save :: very_first_pass = .true.


  call rdfs_get_instant

  allocate(Z_axis(rdfs%N_Z_BINS))
  do i = 1, rdfs%N_Z_BINS; Z_axis(i) = dble(i)/dble(rdfs%N_Z_BINS)*sim_cel(9) ; enddo
  i = int(di_counter)  ;  call char_intN_ch(4,i,ch4)
  open(unit=310,file=trim(path_out)//'rdf_'//trim(continue_job%field1%ch)//'_'//trim(ch4),recl=5000 )
if (l_details_rdf_CTRL) then
  open(unit=311,file=trim(path_out)//'rdf_par_'//trim(continue_job%field1%ch)//'_'//trim(ch4),recl=5000 )
  open(unit=312,file=trim(path_out)//'rdf_perp_'//trim(continue_job%field1%ch)//'_'//trim(ch4),recl=5000 )
endif

   do NB = 1, rdfs%N_Z_BINS

          do i = 0, N_BIN_rdf
           write(310, '(F8.4,1X)', ADVANCE='NO') Z_axis(NB)
           write(310, '(F8.4,1X)', ADVANCE='NO') (dble(i)+0.5d0)*BIN_rdf
           do j = 1, rdfs%N_pairs
             write(310,'(F10.6,1X)',ADVANCE='NO') RA_gr_counters(i,NB,j)
           enddo ! 1, N_pairs_A_A_for_rdf
             write(310,'(A1)',ADVANCE='YES') ' '
          enddo ! i = 1, N_BIN_rdf

if (l_details_rdf_CTRL) then
          do i = 0, N_BIN_rdf
           write(311, '(F8.4,1X)', ADVANCE='NO') Z_axis(NB)
           write(311, '(F8.4,1X)', ADVANCE='NO') (dble(i)+0.5d0)*BIN_rdf
           do j = 1, rdfs%N_pairs
             write(311,'(F10.6,1X)',ADVANCE='NO') RA_gr_counters_par(i,NB,j)
           enddo ! 1, N_pairs_A_A_for_rdf
             write(311,'(A1)',ADVANCE='YES') ' '
          enddo ! i = 1, N_BIN_rdf_par

          do i = 0, N_BIN_rdf
           write(312, '(F8.4,1X)', ADVANCE='NO') Z_axis(NB)
           write(312, '(F8.4,1X)', ADVANCE='NO') (dble(i)+0.5d0)*BIN_rdf
           do j = 1, rdfs%N_pairs
             write(312,'(F10.6,1X)',ADVANCE='NO') RA_gr_counters_perp(i,NB,j)
           enddo ! 1, N_pairs_A_A_for_rdf
             write(312,'(A1)',ADVANCE='YES') ' '
          enddo ! i = 1, N_BIN_rdf_perp
endif!
    enddo ! NB = 1, MAX_BINS_Z_PROFFILE_3

 

if (rdfs%N_Z_BINS > 1) then
     open(unit=333,file=trim(path_out)//'rdf_ALL_'//trim(continue_job%field1%ch)//'_'//trim(ch4),recl=5000 )
          do i = 0, N_BIN_rdf
           write(333, '(F8.4,1X)', ADVANCE='NO') (dble(i)+0.5d0)*BIN_rdf
           do j = 1, rdfs%N_pairs
             write(333,'(F10.6,1X)',ADVANCE='NO') RA_gr(i,j)
           enddo ! 1, N_pairs_A_A_for_rdf
if (l_details_rdf_CTRL) then
           do j = 1, rdfs%N_pairs
             write(333,'(F10.6,1X)',ADVANCE='NO') RA_gr_par(i,j)
           enddo ! 1, N_pairs_A_A_f
            do j = 1, rdfs%N_pairs
             write(333,'(F10.6,1X)',ADVANCE='NO') RA_gr_perp(i,j)
           enddo ! 1
endif !
           write(333,'(A1)',ADVANCE='YES') ' '
          enddo ! i = 1, N_BIN_rdf
     

    close(333) 
endif

 
!   PRINT THEM

 deallocate(Z_axis)
 close(310)
if (l_details_rdf_CTRL) then
 close(311)
 close(312)
endif 
 end subroutine rdfs_print

 subroutine rdfs_get_instant
    use math_constants, only : Pi,Pi_V
    use rdfs_data
    use cut_off_data, only : cut_off
    real(8) BIN_rdf3, rr, aa_counts, d_g,d_perp,d_par,cut2
    integer i,j,k

      if (rdfs%N_Z_BINS > 1) then
         do j = 1, rdfs%N_pairs
         do i = 1, N_BIN_rdf
           gr(i,j) = gr(i,j) + sum(gr_counters(i,:, j))
         enddo
         gr_all_bins(j) = gr_all_bins(j) + sum(ALL_gr_counters(:,j))
         enddo
       if (l_details_rdf_CTRL) then
         do j = 1, rdfs%N_pairs
         do i = 1, N_BIN_rdf
           gr_par(i,j) = gr_par(i,j) + sum(gr_counters_par(i,:, j))
           gr_perp(i,j) = gr_perp(i,j) + sum(gr_counters_perp(i,:, j))
         enddo
           gr_all_bins_par(j) = gr_all_bins_par(j) + sum(ALL_gr_counters_par(:,j))
           gr_all_bins_perp(j) = gr_all_bins_perp(j) + sum(ALL_gr_counters_perp(:,j))
         enddo
       endif
      endif


      BIN_rdf3 = BIN_rdf**3
      d_g = (dble(N_BIN_rdf))**3
      d_par = 4.0d0/3.0d0*Cut_off**3/BIN_rdf*0.5d0
      d_perp = d_par
      cut2 = cut_off**2

    do j = 1, rdfs%N_Z_BINS ; do k = 1,rdfs%N_pairs

        if (ALL_gr_counters(j,k).eq.0.0d0) then
          RA_gr_counters(0:N_BIN_rdf,j,k) = 0.0d0
        else
          aa_counts = d_g/ALL_gr_counters(j,k)
          do i = 0,N_BIN_rdf-1
            RA_gr_counters(i,j,k) = gr_counters(i,j,k)*aa_counts/(dble(i+1)**3-dble(i)**3)
          enddo
        endif

if (l_details_rdf_CTRL) then
     if (ALL_gr_counters_par(j,k)==0.0d0) then
         RA_gr_counters_par(0:N_BIN_rdf,j,k) = 0.0d0
     else
        aa_counts = d_par/(ALL_gr_counters_par(j,k))
        do i = 0,N_BIN_rdf-1
         rr = (BIN_rdf*(dble(i)+0.5d0))**2 
         RA_gr_counters_par(i,j,k) = gr_counters_par(i,j,k)*aa_counts/(cut2-rr)
         enddo
     endif

      if (ALL_gr_counters_perp(j,k)==0.0d0 ) then
          RA_gr_counters_perp(0:N_BIN_rdf,j,k) = 0.0d0
      else
         aa_counts = d_perp/(ALL_gr_counters_perp(j,k))
         do i = 0,N_BIN_rdf-1
           rr= (BIN_rdf*(dble(i)+0.5d0))
           RA_gr_counters_perp(i,j,k) = gr_counters_perp(i,j,k)*aa_counts/&
            (dsqrt(cut2-rr**2)*(2.0d0*rr+BIN_rdf))
          enddo
      endif
endif ! details


    enddo ; enddo 

    ! Now I do the 'global' rdf
if (rdfs%N_Z_BINS > 1) then 
    do k = 1,rdfs%N_pairs

        if (gr_all_bins(k).eq.0.0d0) then
          RA_gr(0:N_BIN_rdf,k) = 0.0d0
        else
          aa_counts = d_g/gr_all_bins(k)
          do i = 0,N_BIN_rdf-1
            RA_gr(i,k) = gr(i,k)*aa_counts/(dble(i+1)**3-dble(i)**3)
          enddo
        endif

if (l_details_rdf_CTRL) then
     if (gr_all_bins_par(k)==0.0d0) then
         RA_gr_par(0:N_BIN_rdf,k) = 0.0d0
     else
        aa_counts = d_par/(gr_all_bins_par(k))
        do i = 0,N_BIN_rdf-1
         rr = (BIN_rdf*(dble(i)+0.5d0))**2
         RA_gr_par(i,k) = gr_par(i,k)*aa_counts/(cut2-rr)
         enddo
     endif

      if (gr_all_bins_perp(k)==0.0d0 ) then
          RA_gr_perp(0:N_BIN_rdf,k) = 0.0d0
      else
         aa_counts = d_perp/(gr_all_bins_perp(k))
         do i = 0,N_BIN_rdf-1
           rr= (BIN_rdf*(dble(i)+0.5d0))
           RA_gr_perp(i,k) = gr_perp(i,k)*aa_counts/&
            (dsqrt(cut2-rr**2)*(2.0d0*rr+BIN_rdf))
          enddo
      endif
endif ! details


    enddo ; 
endif ! if do globals ! do globals if 
  end subroutine rdfs_get_instant

END MODULE rdfs_module


