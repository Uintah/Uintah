 module allocate_them

 public :: smpe_alloc
 public :: smpe_DEalloc
 public :: ALL_atoms_alloc
  contains

 subroutine ALL_atoms_alloc
   use ALL_atoms_data
   implicit none
   integer Na
   Na = Natoms
   allocate(atom_in_which_molecule(Na))
   allocate(atom_in_which_type_molecule(Na))
   allocate(i_type_atom(Na))
   allocate(i_Style_atom(Na))
   allocate(contrains_per_atom(Na))
   allocate(ttx(Na),tty(Na),ttz(Na))
   allocate(xxx(Na),yyy(Na),zzz(Na))
   allocate(xx(Na),yy(Na),zz(Na))
   allocate(vxx(Na),vyy(Na),vzz(Na)) ; vxx=0.0d0; vyy=0.0d0; vzz=0.0d0
   allocate(axx(Na),ayy(Na),azz(Na)) ; axx=0.0d0; ayy=0.0d0; azz=0.0d0
   allocate(fxx(Na),fyy(Na),fzz(Na)) ; fxx=0.0d0; fyy=0.0d0; fzz=0.0d0
   allocate(all_atoms_mass(Na), all_atoms_massinv(Na))
   allocate(all_p_charges(Na)) ; all_p_charges=0.0d0
   allocate(all_g_charges(Na)) ; all_g_charges=0.0d0
   allocate(all_charges(Na)) ; all_charges=0.0d0
   allocate(all_dipoles_xx(Na)); all_dipoles_xx=0.0d0
   allocate(all_dipoles_yy(Na)); all_dipoles_yy=0.0d0
   allocate(all_dipoles_zz(Na)); all_dipoles_zz=0.0d0

 end subroutine ALL_atoms_alloc


  subroutine smpe_alloc
     use variables_smpe
     implicit none
     NFFT = nfftx * nffty * nfftz
     h_cut_z = nfftz/2
     if (.not.allocated(key1)) then 
       allocate(key1(nfftx))
     endif
     key1=0.0d0
     if (.not.allocated(key2)) then
       allocate(key2(nffty))
     endif
     key2=0.0d0;
     if (.not.allocated(key3)) then
       allocate(key3(nfftz))
     endif
     key3=0.0d0; 
     if (.not.(allocated(ww1_Re))) then 
        allocate(ww1_Re(nfftx))
     endif
     ww1_Re=0.0d0
     if (.not.(allocated(ww1_Im))) then
        allocate(ww1_Im(nfftx))
     endif
     ww1_Im=0.0d0
     if (.not.(allocated(ww2_Re))) then
         allocate(ww2_Re(nffty)) 
     endif
     ww2_Re = 0.0d0
    if (.not.(allocated(ww2_Im))) then
        allocate(ww2_Im(nffty))
     endif
     ww2_Im=0.0d0
     if (.not.(allocated(ww3_Re))) then
        allocate(ww3_Re(nfftz))
     endif
     ww3_Re=0.0d0
     if (.not.(allocated(ww3_Im))) then
        allocate(ww3_Im(nfftz))
     endif
     ww3_Im=0.0d0
    if (.not.(allocated(ww1))) then
        allocate(ww1(nfftx))
     endif
     ww1=(0.0d0,0.0d0)
     if (.not.(allocated(ww2))) then
        allocate(ww2(nffty))
     endif
     ww2=(0.0d0,0.0d0)
     if (.not.(allocated(ww3))) then
         allocate(ww3(nfftz))
     endif
     ww3 = (0.0d0,0.0d0)
     if (.not.(allocated(qqq1))) then
        allocate(qqq1(nfftx*nffty*nfftz))
     endif
     qqq1=(0.0d0,0.0d0)



     if (.not.(allocated(spline2_CMPLX_xx))) then 
         allocate(spline2_CMPLX_xx(nfftx))
     endif
     spline2_CMPLX_xx=0.0d0
     if (.not.(allocated(spline2_CMPLX_yy))) then
         allocate(spline2_CMPLX_yy(nffty))
     endif
     spline2_CMPLX_yy=0.0d0
     if (.not.(allocated(spline2_CMPLX_zz))) then
         allocate(spline2_CMPLX_zz(nfftz))
     endif
     spline2_CMPLX_zz=0.0d0 
     if (.not.allocated(qqq1_Re)) then 
           allocate(qqq1_Re(NFFT)) ;
     endif
      qqq1_Re = 0.0d0

  end subroutine smpe_alloc

  subroutine smpe_DEalloc
  use variables_smpe

    if (allocated(key1)) deallocate(key1)
    if (allocated(key2)) deallocate(key2)
    if (allocated(key3)) deallocate(key3)

    if (allocated(ww1_Re)) deallocate(ww1_Re)
    if (allocated(ww1_Im)) deallocate(ww1_Im)
    if (allocated(ww2_Re)) deallocate(ww2_Re)
    if (allocated(ww2_Im)) deallocate(ww2_Im)
    if (allocated(ww3_Re)) deallocate(ww3_Re)
    if (allocated(ww3_Im)) deallocate(ww3_Im)  
    if (allocated(ww1)) deallocate(ww1)
    if (allocated(ww2)) deallocate(ww2)  
    if (allocated(ww3)) deallocate(ww3)  
    if (allocated(qqq1)) deallocate(qqq1) 
    if (allocated(qqq2)) deallocate(qqq2)   
    if (allocated(qqq1_Re)) deallocate(qqq1_Re) 
    if (allocated(qqq1_Im)) deallocate(qqq1_Im)
    if (allocated(qqq2_Re)) deallocate(qqq2_Re) 
    if (allocated(qqq2_Im)) deallocate(qqq2_Im)
       
    if (allocated(spline2_CMPLX_xx)) deallocate(spline2_CMPLX_xx)
    if (allocated(spline2_CMPLX_yy)) deallocate(spline2_CMPLX_yy)
    if (allocated(spline2_CMPLX_zz)) deallocate(spline2_CMPLX_zz)


  end subroutine smpe_DEalloc

 end module allocate_them
