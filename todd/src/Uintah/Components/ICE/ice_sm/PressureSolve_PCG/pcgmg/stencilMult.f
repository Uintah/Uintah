      subroutine stencilMult( x, y, ap, ae, aw, an, as, m, n )

c  $Id$

      implicit none

      integer m
      integer n

      double precision x(m*n)
      double precision y(m*n)

      double precision ap(m,n)
      double precision ae(m,n)
      double precision aw(m,n)
      double precision an(m,n)
      double precision as(m,n)

      double precision zero,       one
      parameter      ( zero=0.0d0, one=1.0d0 )

      include 'space.h'
      include 'grids.h'
      include 'variables.h'

      call copy_to_gridfcn( x, m*n, 
     &                      u(g(0)), mg_dlo(1,0), mg_dhi(1,0),
     &                               mg_vlo(1,0), mg_vhi(1,0) )

      call setbc( u(g(0)), mg_dlo(1,0), mg_dhi(1,0),
     &                     mg_vlo(1,0), mg_vhi(1,0) )

      call gemv( f(g(0)), mg_dlo(1,0), mg_dhi(1,0),
     &           u(g(0)), mg_dlo(1,0), mg_dhi(1,0), 
     &           r(g(0)), mg_dlo(1,0), mg_dhi(1,0), 
     &                    mg_vlo(1,0), mg_vhi(1,0),
     &           ap, ae, aw, an, as, mg_vlo(1,0), mg_vhi(1,0),
     &                                              zero, one )

      call copy_from_gridfcn( r(g(0)), mg_dlo(1,0), mg_dhi(1,0), 
     &                                 mg_vlo(1,0), mg_vhi(1,0),
     &                        y, m*n )

      return
      end

