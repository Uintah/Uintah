      subroutine psol( b, z, m, n )

c  $Id$

      implicit none

c  Purpose:  Approximately solve Mz = b.  
c            This version implements one iteration of a multigrid V-cycle.
c            V(1,1) cycles are used for now, generalize later.

c  Arguments:
c    Input:   b     - vector to be preconditioned
c             b_dlo - index pair for lower left corner of
c                     defined domain
c             b_dhi - index pair for upper right corner of
c                     defined domain
c             b_vlo - index pair for lower left corner of
c                     valid domain
c             b_vhi - index pair for upper right corner of
c                     valid domain

c    Output:  z     - preconditioned vector
c             z_dlo - index pair for lower left corner of
c                     defined domain
c             z_dhi - index pair for upper right corner of
c                     defined domain
c             z_vlo - index pair for lower left corner of
c                     valid domain
c             z_vhi - index pair for upper right corner of
c                     valid domain

c  Dummy arguments-

      integer m, n

      double precision b(m*n)
      double precision z(m*n)

c  Local variables-

      integer l

c  Parameters-

      double precision zero,       one
      parameter      ( zero=0.0d0, one=1.0d0 )

c  Common blocks -

      include 'space.h'
      include 'grids.h'
      include 'variables.h'
      include 'operators.h'

c  Start of exectuable code-

c  Copy input source term into f.

      call copy_to_gridfcn( b, m*n,
     &                      f(g(0)), mg_dlo(1,0), mg_dhi(1,0),
     &                               mg_vlo(1,0), mg_vhi(1,0) )

c  Initialize solution to zero.

      call setval( u(g(0)), mg_dlo(1,0), mg_dhi(1,0), zero )

c  Descent phase.

      do l = 0, levels-1

c  Pre-smooth.

         call relax( u(g(l)), mg_dlo(1,l), mg_dhi(1,l), 
     &                        mg_vlo(1,l), mg_vhi(1,l),
     &               f(g(l)), mg_dlo(1,l), mg_dhi(1,l),
     &               ap(g(l)), ae(g(l)), aw(g(l)), an(g(l)), as(g(l)),
     &                                     mg_vlo(1,l), mg_vhi(1,l), 1 )

c  New residual.

         call gemv( f(g(l)), mg_dlo(1,l), mg_dhi(1,l),
     &              u(g(l)), mg_dlo(1,l), mg_dhi(1,l), 
     &              r(g(l)), mg_dlo(1,l), mg_dhi(1,l), 
     &                       mg_vlo(1,l), mg_vhi(1,l),
     &              ap(g(l)), ae(g(l)), aw(g(l)), an(g(l)), as(g(l)),
     &                             mg_vlo(1,l), mg_vhi(1,l), one, -one )

c  Restrict.

         call inject( r(g(l)), mg_dlo(1,l), mg_dhi(1,l),
     &                         mg_vlo(1,l), mg_vhi(1,l),
     &                f(g(l+1)), mg_dlo(1,l+1), mg_dhi(1,l+1),
     &                           mg_vlo(1,l+1), mg_vhi(1,l+1) ) 

      end do

c  Solve coarsest grid with nu_coarsegrid sweeps.

         call relax( u(g(levels)), mg_dlo(1,levels), mg_dhi(1,levels), 
     &                             mg_vlo(1,levels), mg_vhi(1,levels),
     &               f(g(levels)), mg_dlo(1,levels), mg_dhi(1,levels),
     &               ap(g(levels)), ae(g(levels)), aw(g(levels)), 
     &               an(g(levels)), as(g(levels)),
     &                          mg_vlo(1,levels), mg_vhi(1,levels), 4 )

c  Ascent phase.

      do l = levels-1, 0, -1

c  Prolong coarse grid correction.

         call interp( u(g(l+1)), mg_dlo(1,l+1), mg_dhi(1,l+1),
     &                           mg_vlo(1,l+1), mg_vhi(1,l+1),
     &                r(g(l)), mg_dlo(1,l), mg_dhi(1,l),
     &                         mg_vlo(1,l), mg_vhi(1,l) )
         call setval( u(g(l+1)), mg_dlo(1,l+1), mg_dhi(1,l+1), zero )

c  Apply correction.

         call gf_axpy( u(g(l)), mg_dlo(1,l), mg_dhi(1,l),
     &                          mg_vlo(1,l), mg_vhi(1,l),
     &                 one,
     &                 r(g(l)), mg_dlo(1,l), mg_dhi(1,l),
     &                          mg_vlo(1,l), mg_vhi(1,l) )
         call setbc( u(g(l)), mg_dlo(1,l), mg_dhi(1,l),
     &                        mg_vlo(1,l), mg_vhi(1,l) )

c  Post-smooth.

         call relax( u(g(l)), mg_dlo(1,l), mg_dhi(1,l), 
     &                        mg_vlo(1,l), mg_vhi(1,l),
     &               f(g(l)), mg_dlo(1,l), mg_dhi(1,l),
     &               ap(g(l)), ae(g(l)), aw(g(l)), an(g(l)), as(g(l)),
     &                                     mg_vlo(1,l), mg_vhi(1,l), 1 )

      end do

c  Copy result into output vector.

      call copy_from_gridfcn( u(g(0)), mg_dlo(1,0), mg_dhi(1,0), 
     &                                 mg_vlo(1,0), mg_vhi(1,0),
     &                        z, m*n )

      return
      end
