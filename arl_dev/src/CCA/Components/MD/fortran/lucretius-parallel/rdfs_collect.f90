

!     In short range forces :
!     do i = 1, Natoms-1
!       do j = i+1,Natoms
!
!       ....
!
!       dx(),dy(),dz(),dr_sq() for atom i 
!      enddo
!     call it here
!     enddo


     module rdfs_collect_module
     implicit none
     public :: rdfs_collect
     private :: get_z_coordinate
     
     contains

     subroutine rdfs_collect(i,neigh)
     use variables_short_pairs, only : dx,dy,dz,dr_sq
     use rdfs_data, only : gr_counters, ALL_gr_counters,gr_counters_perp,ALL_gr_counters_perp,&
                    gr_counters_par,ALL_gr_counters_par,BIN_rdf,BIN_rdf_inv,which_pair_rdf,l_rdf_pair_eval,&
                    l_details_rdf_CTRL,rdfs
     use ALL_atoms_data, only : zz, Natoms, i_type_atom
     use cut_off_data, only : cut_off
     use non_bonded_lists_data, only : list_nonbonded,size_list_nonbonded
     implicit none
     integer, intent(IN) :: i,neigh  !neigh is the size list nonbonded of i
     integer it,jt,i_bin_gr,iaz,jaz,i_pair,j,k,NZ
     real(8) dr,dr_par,dr_perp
     
     NZ = rdfs%N_Z_BINS
     it = i_type_atom(i)
     call get_z_coordinate(zz(i),NZ,iaz)
     
     do k = 1,neigh 
        j = list_nonbonded(i,k) 
        jt = i_type_atom(j)
        if (l_rdf_pair_eval(it,jt)) then   ! eval it
           dr = dsqrt(dr_sq(k))
           if (dr < cut_off) then
              i_pair = which_pair_rdf(it,jt)
              call get_z_coordinate(zz(j),NZ,jaz)
              i_bin_gr = int(dr*BIN_rdf_inv)
              gr_counters(i_bin_gr,iaz,i_pair) = gr_counters(i_bin_gr,iaz,i_pair) + 1.0d0
              gr_counters(i_bin_gr,jaz,i_pair) = gr_counters(i_bin_gr,jaz,i_pair) + 1.0d0
              ALL_gr_counters(iaz,i_pair) = ALL_gr_counters(iaz,i_pair) + 1.0d0
              ALL_gr_counters(jaz,i_pair) = ALL_gr_counters(jaz,i_pair) + 1.0d0

if (l_details_rdf_CTRL) then

              dr_perp = dsqrt(dx(k)*dx(k)+dy(k)*dy(k))
              i_bin_gr = int(dr_perp*BIN_rdf_inv)
              gr_counters_perp(i_bin_gr,iaz,i_pair) = gr_counters_perp(i_bin_gr,iaz,i_pair) + 1.0d0
              gr_counters_perp(i_bin_gr,jaz,i_pair) = gr_counters_perp(i_bin_gr,jaz,i_pair) + 1.0d0
              ALL_gr_counters_perp(iaz,i_pair) = ALL_gr_counters_perp(iaz,i_pair) + 1.0d0
              ALL_gr_counters_perp(jaz,i_pair) = ALL_gr_counters_perp(jaz,i_pair) + 1.0d0

              dr_par = dabs(dz(k))
              i_bin_gr = int(dr_par*BIN_rdf_inv)
              gr_counters_par(i_bin_gr,iaz,i_pair) = gr_counters_par(i_bin_gr,iaz,i_pair) + 1.0d0
              gr_counters_par(i_bin_gr,jaz,i_pair) = gr_counters_par(i_bin_gr,jaz,i_pair) + 1.0d0
              ALL_gr_counters_par(iaz,i_pair) = ALL_gr_counters_par(iaz,i_pair) + 1.0d0
              ALL_gr_counters_par(jaz,i_pair) = ALL_gr_counters_par(jaz,i_pair) + 1.0d0 

endif

           endif ! within cut off
        endif     
     enddo

     end subroutine rdfs_collect


     subroutine get_z_coordinate(z,NBINS,jout)
     use sim_cel_data, only : sim_cel
     implicit none
     real(8), intent(IN) :: z
     integer, intent(IN) :: NBINS
     integer, intent(OUT) :: jout
      jout = INT((z/sim_cel(9) + 0.5d0) * dble(NBINS)) + 1
      jout = min(jout,NBINS)
     end subroutine get_z_coordinate


  end  module rdfs_collect_module
