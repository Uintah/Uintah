
 module MC_moves
 public :: MC_move_1atom

 contains
 subroutine MC_trans_move_1atom(iwhich)
 use basic_MC_data
 use ALL_atoms_data, only : xxx,yyy,zzz,Natoms,xx,yy,zz,l_proceed_kin_atom
 use random_generator_module, only : ran2
 use d_energies_data, only : d_en_pot
 use boundaries, only : periodic_images 
 use energy_driver_module_ENERGY, only : energy_difference_MC_move_1atom_1step
   implicit none
   integer, intent(INOUT) :: iwhich ! which atom will be moved
   real(8) dx,dy,dz, xold,yold,zold, xxold,yyold,zzold
   real(8) P

   if (.not.l_proceed_kin_atom(iwhich)) return ! skip is wall or dummy

   dx = (ran2(i_seed)-0.5d0) * mc_translate_displ_xx ! 
   dy = (ran2(i_seed)-0.5d0) * mc_translate_displ_yy
   dz = (ran2(i_seed)-0.5d0) * mc_translate_displ_zz
   xold = xxx(iwhich) ; xxold = xx(iwhich)
   yold = yyy(iwhich) ; yyold = yy(iwhich)
   zold = zzz(iwhich) ; zzold = zz(iwhich)
   xxx(iwhich) = xxx(iwhich) + dx
   yyy(iwhich) = yyy(iwhich) + dy 
   zzz(iwhich) = zzz(iwhich) + dz
   call energy_difference_MC_move_1atom_1step(iwhich)
   P = dexp(-d_en_pot)
   if ( P < ran2(i_seed) ) then
! REJECT MOVE
     xxx(iwhich) = xold ; xx(iwhich) = xxold ! restore coordinates
     yyy(iwhich) = yold ; yy(iwhich) = yyold
     zzz(iwhich) = zold ; zz(iwhich) = zzold
   else
! ACCEPT MOVE
     xx(iwhich) = xxx(iwhich) ; yy(iwhich) = yyy(iwhich) ; zz(iwhich) = zzz(iwhich)
     call periodic_images(xx(iwhich:iwhich),yy(iwhich:iwhich),zz(iwhich:iwhich))
   endif  
 end subroutine MC_trans_move_1atom
 end module MC_moves
