      subroutine pset( m, n, mgLevels, 
     &                 center, east, west, north, south )

c $Id$

      implicit none

      integer mgLevels
      integer m
      integer n

      double precision center(m,n)
      double precision east(m,n)
      double precision west(m,n)
      double precision north(m,n)
      double precision south(m,n)

      integer i
      integer l

      integer op_lo(2), op_hi(2)

      include 'space.h'
      include 'grids.h'
      include 'variables.h'
      include 'operators.h'

      double precision zero
      parameter      ( zero=0.0d0 )

c  Start of executable code-

      if ( mgLevels .gt. MAX_LEVEL ) then
         print*, ' Not enough levels allocated'
         stop
      endif
      if ((2**mgLevels .ge. m) .or. (2**mgLevels .ge. n)) then
         print*, ' m = ', m, ' mgLevels = ', mgLevels
         print*, ' Too many levels for finest grid'
         stop
      endif

      levels = mgLevels

      call setLevelPtrs( m, n, levels, g )

      if (g(levels) .gt. MG_SIZE) then
        print*, 'Not enough Fortran storage pre-allocated'
        stop 
      endif

      call setLevelShapes( m, n, levels, 
     &                     mg_dlo, mg_dhi, mg_vlo, mg_vhi )

c  Initialize variables on all levels.

      do l = 0, levels
         call setval( u(g(l)), mg_dlo(1,l), mg_dhi(1,l), zero )
         call setval( r(g(l)), mg_dlo(1,l), mg_dhi(1,l), zero )
         call setval( f(g(l)), mg_dlo(1,l), mg_dhi(1,l), zero )
      end do

c  Initialize operators on all levels.

      op_lo(1) = 1
      op_lo(2) = 1
      op_hi(1) = m
      op_hi(2) = n

      call setOperator( center, east, west, north, south, op_lo, op_hi,
     &                  ap(g(0)), 
     &                  ae(g(0)), aw(g(0)), an(g(0)), as(g(0)),
     &                  mg_vlo(1,0), mg_vhi(1,0) )
      do l = 1, levels
         call setCoarserOperator(  
     &                   ap(g(l-1)), 
     &                   ae(g(l-1)), aw(g(l-1)), an(g(l-1)), as(g(l-1)),
     &                   mg_vlo(1,l-1), mg_vhi(1,l-1), 
     &                   ap(g(l)), 
     &                   ae(g(l)), aw(g(l)), an(g(l)), as(g(l)),
     &                   mg_vlo(1,l), mg_vhi(1,l) )
      end do

      return
      end
