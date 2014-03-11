module IO_module
implicit none
contains

subroutine read_parameters(file_name)
use Ewald_data
implicit none
character(*) , intent(IN) :: file_name
open(unit=10,file=trim(file_name))  ! open the input file
read(10,*) i_type_EWALD_CTRL ! 1=smpe 2=fast
read(10,*) nfftx,nffty,nfftz ! fft grid
read(10,*) order_spline_xx,order_spline_yy,order_spline_zz ! order of splines
read(10,*) k_max_x,k_max_y,k_max_z ! k vectors for slow ewald
close(10)
end subroutine read_parameters

subroutine read_config(file_name)
use ALL_atoms_data
use sim_cel_data, only : sim_cel
use boundaries, only : periodic_images
use allocate_them, only : ALL_atoms_alloc
use physical_constants, only : Red_Vacuum_EL_permitivity_4_Pi
implicit none
character(*),intent(IN) :: file_name
integer i
sim_cel = 0.0d0;
 open (unit=10, file=trim(file_name))  ! open the input file
 read(10,*) Natoms ! read the number of atoms
 call ALL_atoms_alloc
 read(10,*) sim_cel(1),sim_cel(5),sim_cel(9) ! read the size of simulation cell
 do i=1,Natoms
   read(10,*) xxx(i),yyy(i),zzz(i),all_p_charges(i) ! read the coordinates of atoms
 enddo
 close(10)
 xx=xxx;yy=yyy;zz=zzz
 !all_p_charges = all_p_charges / dsqrt(Red_Vacuum_EL_permitivity_4_Pi)
 all_g_charges=0.0d0; all_charges = all_p_charges; 
 call periodic_images(xx,yy,zz)
end subroutine read_config

subroutine write_config(file_name)
use ALL_atoms_data
use sim_cel_data, only : sim_cel
use boundaries, only : periodic_images
use allocate_them, only : ALL_atoms_alloc
use physical_constants, only : Red_Vacuum_EL_permitivity_4_Pi
implicit none
character(*),intent(IN) :: file_name
integer i
 xx=xxx;yy=yyy;zz=zzz; call periodic_images(xx,yy,zz)
 open (unit=10, file=trim(file_name),recl=600)  ! 
 write(10,*) Natoms ! read the number of atoms
 write(10,*) sim_cel(1),sim_cel(5),sim_cel(9) ! read the size of simulation cell
 do i=1,Natoms
   write(10,*) xx(i),yy(i),zz(i),all_p_charges(i)*dsqrt(Red_Vacuum_EL_permitivity_4_Pi) ! read the coordinates of atoms
 enddo
 close(10)
end subroutine write_config

end module IO_module
